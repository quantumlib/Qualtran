#ifndef KICKGEN_MEM_UTIL_H
#define KICKGEN_MEM_UTIL_H

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace kickmix {

uint32_t max_u32(const uint32_t *items, size_t num_items);
void memcpy_repeat(void *dst, const void *src, size_t src_len, size_t repetitions);

template <typename T, size_t alignment>
static T *aligned_alloc_else_throw(size_t count) {
    static_assert(alignment > 0, "Alignment must be a power of two.");
    constexpr size_t align_mask = alignment - size_t{1};
    static_assert((alignment & align_mask) == 0, "Alignment must be a power of two.");
    static_assert(alignment <= 1024, "Alignment exceeded 1024.");

    if (count == 0) {
        return nullptr;
    }
    if (count > (SIZE_MAX - alignment) / sizeof(T)) {
        throw std::runtime_error("Failed to allocate.");
    }
    size_t bytes = count * sizeof(T);
    bytes += align_mask;
    bytes &= ~align_mask;

#if defined(_MSC_VER)
    T *result = (T *)_aligned_alloc(alignment, bytes);
#else
    T *result = (T *)std::aligned_alloc(alignment, bytes);
#endif

    if (result == nullptr) {
        throw std::runtime_error("Failed to allocate.");
    }
    return result;
}

template <typename T>
static T *alloc_else_throw(size_t count) {
    return aligned_alloc_else_throw<T, alignof(T)>(count);
}

template <typename T>
inline void free_aligned(T *items) noexcept {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(items);
#endif
}

}  // namespace kickmix

#endif
