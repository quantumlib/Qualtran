#include "kickmix/mem/monotonic_arena.h"

static void dealloc(kickmix::MonotonicArena_Helper &buf) {
    for (const auto &e : buf.frozen_allocations) {
        free(e.data());
    }
    buf.frozen_allocations.clear();
    buf.cur_length_used = 0;
    buf.total_frozen_length_used = 0;
    buf.cur_length_cap = 0;
    if (buf.cur != nullptr) {
        free(buf.cur);
        buf.cur = nullptr;
    }
}

kickmix::MonotonicArena_Helper::MonotonicArena_Helper()
    : frozen_allocations(), total_frozen_length_used(0), cur(nullptr), cur_length_used(0), cur_length_cap(0) {
}

kickmix::MonotonicArena_Helper::MonotonicArena_Helper(MonotonicArena_Helper &&other) noexcept
    : frozen_allocations(std::move(other.frozen_allocations)),
      total_frozen_length_used(other.total_frozen_length_used),
      cur(other.cur),
      cur_length_used(other.cur_length_used),
      cur_length_cap(other.cur_length_cap) {
    other.total_frozen_length_used = 0;
    other.cur = 0;
    other.cur_length_used = 0;
    other.cur_length_cap = 0;
}
kickmix::MonotonicArena_Helper &kickmix::MonotonicArena_Helper::operator=(MonotonicArena_Helper &&other) noexcept {
    if (this == &other) {
        return *this;
    }
    dealloc(*this);
    frozen_allocations = std::move(other.frozen_allocations);
    total_frozen_length_used = std::move(other.total_frozen_length_used);
    cur = std::move(other.cur);
    cur_length_used = std::move(other.cur_length_used);
    cur_length_cap = std::move(other.cur_length_cap);
    other.total_frozen_length_used = 0;
    other.cur = 0;
    other.cur_length_used = 0;
    other.cur_length_cap = 0;
    return *this;
}

kickmix::MonotonicArena_Helper::~MonotonicArena_Helper() {
    dealloc(*this);
}

void kickmix::MonotonicArena_Helper::clear() {
    for (const auto &e : frozen_allocations) {
        free(e.data());
    }
    frozen_allocations.clear();
    cur_length_used = 0;
    total_frozen_length_used = 0;
}

void kickmix::MonotonicArena_Helper::rewind_tail(char *end) {
    if (cur <= end && end <= cur + cur_length_used) {
        cur_length_used = end - cur;
    } else {
        throw std::invalid_argument("rewind_tail was given a pointer outside the current buffer.");
    }
}

char *kickmix::MonotonicArena_Helper::grab_writeable(size_t length, size_t alloc_alignment) {
    // Some aligned_alloc implementations (including macOS) require pointer-sized alignment.
    alloc_alignment = std::max(alloc_alignment, sizeof(void *));
    if (cur_length_used + length >= cur_length_cap) {
        if (cur != nullptr) {
            if (cur_length_used > 0) {
                frozen_allocations.push_back(std::span<char>{cur, cur_length_used});
                total_frozen_length_used += cur_length_used;
            } else {
                free(cur);
            }
        }

        size_t new_size = cur_length_cap;
        new_size = std::max(new_size, alloc_alignment * 4);
        new_size = std::max(new_size, length);
        new_size <<= 1;
        new_size += alloc_alignment - 1;
        new_size &= ~size_t{alloc_alignment - 1};
        cur = (char *)aligned_alloc(alloc_alignment, new_size);
        cur_length_used = 0;
        cur_length_cap = new_size;
        if (cur == nullptr) {
            cur_length_cap = 0;
            throw std::invalid_argument("Failed to allocate");
        }
    }

    char *result = cur + cur_length_used;
    cur_length_used += length;
    return result;
}

void kickmix::MonotonicArena_Helper::memcpy_into(char *dst) const {
    for (const auto &e : frozen_allocations) {
        memcpy(dst, e.data(), e.size());
        dst += e.size();
    }
    if (cur_length_used) {
        memcpy(dst, cur, cur_length_used);
    }
}
