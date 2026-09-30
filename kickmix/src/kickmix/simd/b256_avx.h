#ifndef KICKMIX_SIMD_256_AVX_H
#define KICKMIX_SIMD_256_AVX_H

#include <cstring>
#ifdef __AVX__
#include <immintrin.h>
#endif
#include <ostream>
#include <random>

namespace kickmix {

#ifdef __AVX__
struct alignas(sizeof(uint64_t) * 4) b256_avx {
    __m256i v;

    constexpr static size_t BIT_SIZE = 256;
    constexpr static const char *NAME = "b256_avx";
    bool non_zero() const {
        uint64_t *r = (uint64_t *)&v;
        return r[0] | r[1] | r[2] | r[3];
    }
    static b256_avx random(std::mt19937_64 &rng) {
        return {_mm256_set_epi64x(rng(), rng(), rng(), rng())};
    }
    void randomize(std::mt19937_64 &rng) {
        v = _mm256_set_epi64x(rng(), rng(), rng(), rng());
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
        v = _mm256_set1_epi64x(0);
    }
    inline void clear_to_max() {
        v = _mm256_set1_epi64x(UINT64_MAX);
    }
    inline bool operator==(const b256_avx &other) const {
        return memcmp(&v, &other.v, sizeof(v)) == 0;
    }
    inline b256_avx &operator^=(const b256_avx &other) {
        v = _mm256_xor_si256(v, other.v);
        return *this;
    }
    inline b256_avx &operator&=(const b256_avx &other) {
        v = _mm256_and_si256(v, other.v);
        return *this;
    }
    inline b256_avx &operator|=(const b256_avx &other) {
        v = _mm256_or_si256(v, other.v);
        return *this;
    }
    inline b256_avx operator~() const {
        return {_mm256_xor_si256(v, _mm256_set1_epi8(-1))};
    }
    inline b256_avx operator^(const b256_avx &other) const {
        return {_mm256_xor_si256(v, other.v)};
    }
    inline b256_avx operator&(const b256_avx &other) const {
        return {_mm256_and_si256(v, other.v)};
    }
    inline b256_avx operator|(const b256_avx &other) const {
        return {_mm256_or_si256(v, other.v)};
    }
    inline static b256_avx from_u32_broadcast(uint32_t v) {
        return {_mm256_set1_epi32(std::bit_cast<int32_t>(v))};
    }
    inline b256_avx &u64_iadd(const b256_avx &other) {
        v = _mm256_add_epi64(v, other.v);
        return *this;
    }
    inline b256_avx u64_add(const b256_avx &other) const {
        return {_mm256_add_epi64(v, other.v)};
    }
    inline b256_avx u64_right_shift(size_t shift) const {
        return {_mm256_srli_epi64(v, shift)};
    }
    inline b256_avx u64_left_shift(size_t shift) const {
        return {_mm256_slli_epi64(v, shift)};
    }
    inline b256_avx u32_eq(const b256_avx &other) const {
        return {_mm256_cmpeq_epi32(v, other.v)};
    }
};
inline std::ostream &operator<<(std::ostream &out, const b256_avx &v) {
    for (size_t k = 0; k < sizeof(v) * 8; k++) {
        out << "_1"[v.bit(k)];
    }
    return out;
}
#endif

}  // namespace kickmix

#endif
