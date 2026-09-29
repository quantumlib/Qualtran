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

}  // namespace kickmix

#endif
