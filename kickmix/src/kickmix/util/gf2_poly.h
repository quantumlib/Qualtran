#ifndef _KICKMIX_UTIL_GF2_POLY_H
#define _KICKMIX_UTIL_GF2_POLY_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>

#include "kickmix/util/fixed_width_int.h"

namespace kickmix {

/// The largest binary extension field degree supported by this library.
///
/// Field elements of GF(2^m) for m <= GF2_MAX_DEGREE fit in GF2_MAX_DEGREE bits, and products of two
/// such elements (before reduction) fit in 2*GF2_MAX_DEGREE bits. GF2Poly is sized to hold the
/// unreduced products, so that multiplication never needs to allocate.
constexpr size_t GF2_MAX_DEGREE = 512;

/// A dense polynomial over GF(2), i.e. a bit string, with inline storage.
///
/// Bit k of the polynomial is the coefficient of x^k. All storage is inline (a fixed size array), so
/// GF2Poly is trivially copyable and never heap allocates. This matters because the classical Galois
/// field arithmetic used to synthesize circuits performs millions of these operations, and the
/// allocation traffic of a variable width representation (such as FixedWidthInt) would dominate.
///
/// The capacity is 2*GF2_MAX_DEGREE bits so that the unreduced product of two field elements fits.
struct GF2Poly {
    /// Number of coefficient bits that fit in a GF2Poly.
    static constexpr size_t MAX_BITS = 2 * GF2_MAX_DEGREE;
    /// Number of 64 bit words backing a GF2Poly.
    static constexpr size_t MAX_WORDS = MAX_BITS / 64;

    /// Coefficients, in little endian bit order. words[k / 64] bit (k % 64) is the coefficient of x^k.
    std::array<uint64_t, MAX_WORDS> words;

    /// Creates the zero polynomial.
    inline constexpr GF2Poly() : words{} {
    }

    /// Creates a polynomial whose low 64 coefficients are the bits of `value`.
    static GF2Poly from_u64(uint64_t value);
    /// Creates the monomial x^k. Requires k < MAX_BITS.
    static GF2Poly monomial(size_t k);
    /// Creates a polynomial from the bits of a FixedWidthInt. Requires value.num_bits <= MAX_BITS.
    static GF2Poly from_fixed_width_int(const FixedWidthInt &value);
    /// Parses a polynomial from a hex ("0x1B") or binary ("0b11011") string.
    static GF2Poly from_str(std::string_view text);

    /// Returns the coefficient of x^k, or false if k >= MAX_BITS.
    inline bool bit(size_t k) const {
        if (k >= MAX_BITS) {
            return false;
        }
        return (words[k / 64] >> (k & 63)) & 1;
    }
    /// Sets the coefficient of x^k. Requires k < MAX_BITS.
    void set_bit(size_t k, bool value);
    /// Toggles the coefficient of x^k. Requires k < MAX_BITS.
    void xor_bit(size_t k);

    /// Returns true if every coefficient is zero.
    bool is_zero() const;
    /// Returns the degree of the polynomial, or SIZE_MAX if the polynomial is zero.
    size_t degree() const;
    /// Returns the number of bits needed to write the polynomial, i.e. degree() + 1 (or 0 if zero).
    size_t num_bits_in_use() const;
    /// Returns the number of non-zero coefficients.
    size_t weight() const;

    /// Zeroes every coefficient of x^k for k >= num_bits.
    void truncate(size_t num_bits);

    inline bool operator==(const GF2Poly &other) const {
        return words == other.words;
    }
    inline bool operator!=(const GF2Poly &other) const {
        return words != other.words;
    }

    /// Adds another polynomial into this one (over GF(2), addition is XOR).
    GF2Poly &operator^=(const GF2Poly &other);
    inline GF2Poly operator^(const GF2Poly &other) const {
        GF2Poly result = *this;
        result ^= other;
        return result;
    }

    /// Multiplies by x^shift, discarding coefficients that overflow MAX_BITS.
    GF2Poly &operator<<=(size_t shift);
    /// Divides by x^shift, discarding the remainder.
    GF2Poly &operator>>=(size_t shift);
    inline GF2Poly operator<<(size_t shift) const {
        GF2Poly result = *this;
        result <<= shift;
        return result;
    }
    inline GF2Poly operator>>(size_t shift) const {
        GF2Poly result = *this;
        result >>= shift;
        return result;
    }

    /// XORs `other << shift` into this polynomial. Coefficients overflowing MAX_BITS are discarded.
    void ixor_shifted(const GF2Poly &other, size_t shift);

    /// Returns the carry-less (GF(2) polynomial) product of two polynomials.
    ///
    /// Requires degree(lhs) + degree(rhs) < MAX_BITS, which always holds for products of two
    /// elements of a field of degree at most GF2_MAX_DEGREE.
    static GF2Poly mul(const GF2Poly &lhs, const GF2Poly &rhs);

    /// Reduces this polynomial modulo `modulus`, in place. Requires modulus != 0.
    void imod(const GF2Poly &modulus);

    /// Returns the greatest common divisor of two polynomials.
    static GF2Poly gcd(GF2Poly lhs, GF2Poly rhs);

    /// Returns the inverse of `value` modulo `modulus`, using the extended Euclidean algorithm.
    ///
    /// Returns the zero polynomial if the inverse does not exist (i.e. if gcd(value, modulus) != 1),
    /// which includes the case value == 0.
    static GF2Poly invert_mod(const GF2Poly &value, const GF2Poly &modulus);

    /// Converts to a FixedWidthInt with the given bit count. Requires num_bits <= MAX_BITS.
    FixedWidthInt to_fixed_width_int(size_t num_bits) const;
    /// Returns a hex representation, e.g. "0x11B".
    std::string str() const;
};

std::ostream &operator<<(std::ostream &out, const GF2Poly &value);

}  // namespace kickmix

#endif
