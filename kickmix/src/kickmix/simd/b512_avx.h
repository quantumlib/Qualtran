#ifndef KICKMIX_SIMD_512_AVX_H
#define KICKMIX_SIMD_512_AVX_H

#include <cstring>
#include <ostream>
#include <random>

#ifdef __AVX512F__
#include "immintrin.h"
#endif

#ifdef __AVX512F__
namespace kickmix {

struct alignas(sizeof(uint64_t) * 8) b512_avx {
    __m512i v;

    constexpr static size_t BIT_SIZE = 512;
    constexpr static const char *NAME = "b512_avx";
    bool non_zero() const {
        uint64_t *result = (uint64_t *)&v;
        return result[0] | result[1] | result[2] | result[3] | result[4] | result[5] | result[6] | result[7];
    }
    static b512_avx random(std::mt19937_64 &rng) {
        return {_mm512_set_epi64(rng(), rng(), rng(), rng(), rng(), rng(), rng(), rng())};
    }
    void randomize(std::mt19937_64 &rng) {
        v = _mm512_set_epi64(rng(), rng(), rng(), rng(), rng(), rng(), rng(), rng());
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
        v = _mm512_set1_epi64(0);
    }
    inline void clear_to_max() {
        v = _mm512_set1_epi64(UINT64_MAX);
    }
    inline bool operator==(const b512_avx &other) const {
        return memcmp(&v, &other.v, sizeof(v)) == 0;
    }
    inline b512_avx &operator^=(const b512_avx &other) {
        v = _mm512_xor_si512(v, other.v);
        return *this;
    }
    inline b512_avx &operator&=(const b512_avx &other) {
        v = _mm512_and_si512(v, other.v);
        return *this;
    }
    inline b512_avx &operator|=(const b512_avx &other) {
        v = _mm512_or_si512(v, other.v);
        return *this;
    }
    inline b512_avx operator~() const {
        return {_mm512_xor_si512(v, _mm512_set1_epi8(-1))};
    }
    inline b512_avx operator^(const b512_avx &other) const {
        return {_mm512_xor_si512(v, other.v)};
    }
    inline b512_avx operator&(const b512_avx &other) const {
        return {_mm512_and_si512(v, other.v)};
    }
    inline b512_avx operator|(const b512_avx &other) const {
        return {_mm512_or_si512(v, other.v)};
    }
    inline static b512_avx from_u32_broadcast(uint32_t v) {
        return {_mm512_set1_epi32(std::bit_cast<int32_t>(v))};
    }
    inline b512_avx &u64_iadd(const b512_avx &other) {
        v = _mm512_add_epi64(v, other.v);
        return *this;
    }
    inline b512_avx u64_add(const b512_avx &other) const {
        return {_mm512_add_epi64(v, other.v)};
    }
    inline b512_avx u64_right_shift(size_t shift) const {
        return {_mm512_srli_epi64(v, shift)};
    }
    inline b512_avx u64_left_shift(size_t shift) const {
        return {_mm512_slli_epi64(v, shift)};
    }
    inline b512_avx u32_eq(const b512_avx &other) const {
        return {_mm512_movm_epi32(_mm512_cmpeq_epi32_mask(v, other.v))};
    }
};
inline std::ostream &operator<<(std::ostream &out, const b512_avx &v) {
    for (size_t k = 0; k < sizeof(v) * 8; k++) {
        out << "_1"[v.bit(k)];
    }
    return out;
}

}  // namespace kickmix

#endif
#endif
