#ifndef KICKMIX_UTIL_WORD_OPS_H
#define KICKMIX_UTIL_WORD_OPS_H

#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace kickmix {

/// Adds two words and a carry bit, returning the carry out.
inline bool add_carry_u64(bool carry, uint64_t a, uint64_t b, unsigned long long *out) {
#if defined(__x86_64__) || defined(_M_X64)
    return _addcarry_u64(carry, a, b, out);
#else
    uint64_t sum = a + b;
    *out = sum + carry;
    return sum < a || *out < sum;
#endif
}

/// Subtracts a word and a borrow bit, returning the borrow out.
inline bool sub_borrow_u64(bool borrow, uint64_t a, uint64_t b, unsigned long long *out) {
#if defined(__x86_64__) || defined(_M_X64)
    return _subborrow_u64(borrow, a, b, out);
#else
    uint64_t difference = a - b;
    *out = difference - borrow;
    return a < b || difference < static_cast<uint64_t>(borrow);
#endif
}

/// Packs the masked bits into the low bits of the result, preserving their order.
inline uint64_t extract_bits_u64(uint64_t value, uint64_t mask) {
#if defined(__BMI2__) && (defined(__x86_64__) || defined(_M_X64))
    return _pext_u64(value, mask);
#else
    uint64_t result = 0;
    uint64_t output_bit = 1;
    while (mask) {
        uint64_t input_bit = mask & (0 - mask);
        if (value & input_bit) {
            result |= output_bit;
        }
        mask &= mask - 1;
        output_bit <<= 1;
    }
    return result;
#endif
}

/// Computes factor1 * factor2 + offset1 + offset2 as a 128-bit unsigned integer.
inline void full_fma64(
    uint64_t factor1, uint64_t factor2, uint64_t offset1, uint64_t offset2, uint64_t *out_low, uint64_t *out_high) {
#if defined(__SIZEOF_INT128__)
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
    auto result = factor1 * (unsigned __int128)factor2 + (unsigned __int128)offset1 + offset2;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
    *out_high = static_cast<uint64_t>(result >> 64);
    *out_low = static_cast<uint64_t>(result);
#else
    uint64_t a_lo = factor1 & 0xFFFFFFFFULL;
    uint64_t a_hi = factor1 >> 32;
    uint64_t b_lo = factor2 & 0xFFFFFFFFULL;
    uint64_t b_hi = factor2 >> 32;

    uint64_t p0 = a_lo * b_lo;
    uint64_t p1 = a_lo * b_hi;
    uint64_t p2 = a_hi * b_lo;
    uint64_t p3 = a_hi * b_hi;

    uint64_t mid = (p0 >> 32) + (p1 & 0xFFFFFFFFULL) + (p2 & 0xFFFFFFFFULL);
    uint64_t lo = (p0 & 0xFFFFFFFFULL) | (mid << 32);
    uint64_t hi = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);

    unsigned long long sum1 = 0;
    bool c1 = add_carry_u64(false, lo, offset1, &sum1);
    unsigned long long sum2 = 0;
    bool c2 = add_carry_u64(false, sum1, offset2, &sum2);
    *out_low = sum2;
    *out_high = hi + c1 + c2;
#endif
}

}  // namespace kickmix

#endif
