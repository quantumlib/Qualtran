#ifndef KICKMIX_SIMD_128_SSE_H
#define KICKMIX_SIMD_128_SSE_H

#include <cstring>
#include <ostream>
#include <random>

#ifdef __SSE__
#include "immintrin.h"
#endif

namespace kickmix {

#ifdef __SSE__
struct alignas(sizeof(uint64_t) * 2) b128_sse {
    __m128i v;

    constexpr static size_t BIT_SIZE = 128;
    constexpr static const char *NAME = "b128_sse";
    bool non_zero() const {
        uint64_t *r = (uint64_t *)&v;
        return r[0] | r[1];
    }
    static b128_sse random(std::mt19937_64 &rng) {
        return {_mm_set_epi64x(rng(), rng())};
    }
    void randomize(std::mt19937_64 &rng) {
        v = _mm_set_epi64x(rng(), rng());
    }
    inline uint64_t u64(size_t k) const {
        return ((uint64_t *)&v)[k];
    }
    inline uint32_t u32(size_t k) const {
        return ((uint32_t *)&v)[k];
    }
    inline bool bit(size_t k) const {
        return (u64(k / 64) >> (k % 64)) & 1;
    }
    inline void set_bit(size_t k, bool new_value) {
        size_t q = k / 64;
        size_t r = k % 64;
        uint64_t t = ((uint64_t *)&v)[q];
        t &= ~(uint64_t{1} << r);
        t |= (uint64_t)new_value << r;
        ((uint64_t *)&v)[q] = t;
    }
    inline void clear_to_zero() {
        v = _mm_set1_epi64x(0);
    }
    inline void clear_to_max() {
        v = _mm_set1_epi64x(UINT64_MAX);
    }
    inline bool operator==(const b128_sse &other) const {
        uint64_t a[2];
        uint64_t b[2];
        memcpy(a, &v, sizeof(a));
        memcpy(b, &other.v, sizeof(b));
        return a[0] == b[0] && a[1] == b[1];
    }
    inline b128_sse &operator^=(const b128_sse &other) {
        v = _mm_xor_si128(v, other.v);
        return *this;
    }
    inline b128_sse &operator&=(const b128_sse &other) {
        v = _mm_and_si128(v, other.v);
        return *this;
    }
    inline b128_sse &operator|=(const b128_sse &other) {
        v = _mm_or_si128(v, other.v);
        return *this;
    }
    inline b128_sse operator~() const {
        return {_mm_xor_si128(v, _mm_set1_epi8(-1))};
    }
    inline b128_sse operator^(const b128_sse &other) const {
        return {_mm_xor_si128(v, other.v)};
    }
    inline b128_sse operator&(const b128_sse &other) const {
        return {_mm_and_si128(v, other.v)};
    }
    inline b128_sse operator|(const b128_sse &other) const {
        return {_mm_or_si128(v, other.v)};
    }
    inline static b128_sse from_u32_broadcast(uint32_t v) {
        return {_mm_set1_epi32(std::bit_cast<int32_t>(v))};
    }
    inline b128_sse &u64_iadd(const b128_sse &other) {
        v = _mm_add_epi64(v, other.v);
        return *this;
    }
    inline b128_sse u64_add(const b128_sse &other) const {
        return {_mm_add_epi64(v, other.v)};
    }
    inline b128_sse u64_right_shift(size_t shift) const {
        return {_mm_srli_epi64(v, shift)};
    }
    inline b128_sse u64_left_shift(size_t shift) const {
        return {_mm_slli_epi64(v, shift)};
    }
    inline b128_sse u32_eq(const b128_sse &other) const {
        return {_mm_cmpeq_epi32(v, other.v)};
    }
};
inline std::ostream &operator<<(std::ostream &out, const b128_sse &v) {
    for (size_t k = 0; k < sizeof(v) * 8; k++) {
        out << "_1"[v.bit(k)];
    }
    return out;
}
#endif

}  // namespace kickmix

#endif
