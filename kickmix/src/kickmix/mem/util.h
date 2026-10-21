#ifndef KICKGEN_MEM_UTIL_H
#define KICKGEN_MEM_UTIL_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#if defined(_MSC_VER) || defined(_WIN32)
#include <malloc.h>
#endif

namespace kickmix {

uint32_t max_u32(const uint32_t *items, size_t num_items);
void memcpy_repeat(void *dst, const void *src, size_t src_len, size_t repetitions);

template <typename T, size_t alignment>
static T *aligned_alloc_else_throw(size_t count) {
    static_assert(alignment > 0, "Alignment must be a power of two.");
    static_assert((alignment & (alignment - size_t{1})) == 0, "Alignment must be a power of two.");
    static_assert(alignment <= 1024, "Alignment exceeded 1024.");
    constexpr size_t eff_alignment = alignment < sizeof(void *) ? sizeof(void *) : alignment;
    constexpr size_t align_mask = eff_alignment - size_t{1};

    if (count == 0) {
        return nullptr;
    }
    if (count > (SIZE_MAX - eff_alignment) / sizeof(T)) {
        throw std::runtime_error("Failed to allocate.");
    }
    size_t bytes = count * sizeof(T);
    bytes += align_mask;
    bytes &= ~align_mask;

#if defined(_MSC_VER) || defined(_WIN32)
    T *result = (T *)_aligned_malloc(bytes, eff_alignment);
#elif defined(__APPLE__)
    void *raw_result = nullptr;
    if (posix_memalign(&raw_result, eff_alignment, bytes) != 0) {
        raw_result = nullptr;
    }
    T *result = (T *)raw_result;
#else
    T *result = (T *)std::aligned_alloc(eff_alignment, bytes);
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
#if defined(_MSC_VER) || defined(_WIN32)
    _aligned_free((void *)items);
#else
    free((void *)items);
#endif
}

}  // namespace kickmix

#endif
