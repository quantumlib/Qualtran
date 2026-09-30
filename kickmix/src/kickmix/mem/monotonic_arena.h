#ifndef KICKGEN_MEM_MONOTONIC_ARENA_H
#define KICKGEN_MEM_MONOTONIC_ARENA_H

#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

#include "stride_span.h"

namespace kickmix {

struct MonotonicArena_Helper {
    std::vector<std::span<char>> frozen_allocations;
    size_t total_frozen_length_used;
    char *cur;
    size_t cur_length_used;
    size_t cur_length_cap;

    MonotonicArena_Helper();
    MonotonicArena_Helper(MonotonicArena_Helper &&other) noexcept;
    MonotonicArena_Helper &operator=(MonotonicArena_Helper &&other) noexcept;
    MonotonicArena_Helper(const MonotonicArena_Helper &) = delete;
    MonotonicArena_Helper &operator=(const MonotonicArena_Helper &) = delete;
    ~MonotonicArena_Helper();

    void clear();
    void rewind_tail(char *end);
    char *grab_writeable(size_t length, size_t alloc_alignment);
    void memcpy_into(char *dst) const;
};

/// A collection of items with support for efficient bulk appending.
///
/// Internally, the collection allocates chunks with exponentially increasing
/// sizes. Within each chunk it uses a bump allocator. A key feature is the
/// ability to give back the tail of the last allocation, allowing items to
/// be appended one by one without performing a capacity check for each item.
///
/// Example:
///     MonotonicArena<size_t> arena;
///     arena.push_back(5);
///
///     // Instead of `reserve`, the arena has `grab_writeable+rewind_tail`:
///     size_t *out = arena.grab_writeable(100);
///     for (size_t k = 0; k < 100; k++) {
///         if (k % 7) {  // Just an arbitrary predicate.
///             *out++ = k;
///         }
///     }
///     arena.rewind_tail(out);  // Return unused memory for next append.
///
///     arena.push_back(3);
template <typename T, size_t alignment>
struct MonotonicArena {
   private:
    static_assert((alignment & (alignment - 1)) == 0);
    static_assert(alignment > 0);
    static_assert(alignment >= sizeof(T));
    static_assert(alignment % sizeof(T) == 0);
    MonotonicArena_Helper content;

   public:
    /// The total number of items in the arena.
    inline size_t size() const {
        return (content.total_frozen_length_used + content.cur_length_used) / sizeof(T);
    }

    /// Clears all items out of the arena.
    ///
    /// This doesn't deallocate the largest chunk of memory used by the arena,
    /// but it does deallocate all smaller chunks.
    inline void clear() {
        content.clear();
    }

    /// Drops items from back of the arena.
    ///
    /// This method is intended to be used in combination with `grab_writeable` in
    /// order to efficiently reserve space for appending items without doing a
    /// capacity check for each append. If `p is the pointer returned by
    /// `grab_writeable(n)`, then the pointer `q` given to this method should
    /// satisfy `p <= q && q <= p + n`. Otherwise undefined behavior occurs.
    inline void rewind_tail(T *end) {
        content.rewind_tail((char *)end);
    }

    /// Allocates enough memory for the given number of items and returns a pointer
    /// to the beginning of the range. If the caller doesn't end up using the entire
    /// range, they MUST return the unused tail by using `arena.rewind_tail(end)`.
    /// Otherwise iterating items in the arena will iterate over the unused items.
    inline T *grab_writeable(size_t count) {
        return (T *)content.grab_writeable(count * sizeof(T), alignment);
    }

    /// Copies all items in the arena to the given destination buffer.
    ///
    /// The number of items copied is `arena.size()`. It's the caller's responsibility
    /// to ensure the destination buffer is large enough.
    inline void memcpy_into(T *dst) const {
        content.memcpy_into((char *)dst);
    }

    /// Calls a callback with each std::span<T> of items in the arena.
    template <typename TCallback>
    void iter_spans(TCallback callback) const {
        for (const auto &e : content.frozen_allocations) {
            callback(std::span<T>{(T *)e.data(), e.size() / sizeof(T)});
        }
        if (content.cur_length_used) {
            callback(std::span<T>{(T *)content.cur, content.cur_length_used / sizeof(T)});
        }
    }

    /// Appends a single item to the end of the arena.
    ///
    /// If the current chunk has run out, this method will allocate a new one.
    /// If this method is being called in a loop, it's recommended to use
    /// `grab_writeable` instead, to reduce the number of capacity checks.
    inline void push_back(T val) {
        *grab_writeable(1) = val;
    }

    /// Repeatedly appends an item to the end of the arena.
    void push_back_repeat(T val, size_t repetitions) {
        if constexpr (sizeof(T) == 1) {
            if (repetitions > 0) {
                memset(grab_writeable(repetitions), (int)val, repetitions);
            }
        } else {
            auto r = grab_writeable(repetitions);
            for (size_t k = 0; k < repetitions; k++) {
                r[k] = val;
            }
        }
    }

    /// Appends multiple items to the end of the arena.
    void push_back_many(std::span<const T> vals) {
        if (!vals.empty()) {
            memcpy(grab_writeable(vals.size()), vals.data(), vals.size() * sizeof(T));
        }
    }

    /// Appends multiple items to the end of the arena.
    void push_back_many(const std::vector<T> vals) {
        if (!vals.empty()) {
            memcpy(grab_writeable(vals.size()), vals.data(), vals.size() * sizeof(T));
        }
    }

    /// Appends multiple items to the end of the arena.
    void push_back_many(stride_span<const T> vals) {
        if (vals.stride == 1 && !vals.empty()) {
            memcpy(grab_writeable(vals.size()), vals.data(), vals.size() * sizeof(T));
        } else {
            T *out = grab_writeable(vals.size());
            for (size_t k = 0; k < vals.size(); k++) {
                *out++ = vals[k];
            }
        }
    }

    /// Appends multiple items, in reverse order, to the end of the arena.
    void push_back_many_reversed(std::span<const T> vals) {
        T *out = grab_writeable(vals.size());
        for (size_t k = vals.size(); k--;) {
            *out++ = vals[k];
        }
    }

    /// Appends all items from the given arena to the end of this arena.
    void push_back_many(const MonotonicArena<T, alignment> &other) {
        other.iter_spans([&](std::span<const T> span) {
            push_back_many(span);
        });
    }

    /// Appends all items from the given arena, in reversed order, to the end of this arena.
    void push_back_many_reversed(const MonotonicArena<T, alignment> &other) {
        push_back_many_reversed(std::span<const T>{(T *)other.content.cur, other.content.cur_length_used / sizeof(T)});
        for (size_t k = other.content.frozen_allocations.size(); k--;) {
            const auto &e = other.content.frozen_allocations[k];
            push_back_many_reversed(std::span<const T>{(T *)e.data(), e.size() / sizeof(T)});
        }
    }
};

}  // namespace kickmix

#endif
