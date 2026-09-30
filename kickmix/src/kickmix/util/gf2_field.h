#ifndef _KICKMIX_UTIL_GF2_FIELD_H
#define _KICKMIX_UTIL_GF2_FIELD_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "kickmix/util/fixed_width_int.h"
#include "kickmix/util/gf2_matrix.h"
#include "kickmix/util/gf2_poly.h"

namespace kickmix {

/// Arithmetic in the binary extension field GF(2^m), for 1 <= m <= GF2_MAX_DEGREE.
///
/// Elements are polynomials of degree less than m over GF(2), represented as GF2Poly values whose
/// bit k is the coefficient of x^k. Addition is XOR, and multiplication is polynomial multiplication
/// reduced modulo an irreducible polynomial of degree m.
///
/// The reduction polynomial is chosen to have low weight (a trinomial when one exists, otherwise a
/// pentanomial), which lets `reduce` clear each high coefficient with a handful of bit flips.
struct GF2Field {
    /// Creates GF(2^m) using gf2_default_irreducible_poly(degree).
    explicit GF2Field(size_t degree);

    /// Creates GF(2^m) using an explicit reduction polynomial.
    ///
    /// The polynomial must be irreducible and have degree exactly `degree`. As a convenience the
    /// x^degree term may be omitted from `irreducible_poly`, in which case it is implied.
    ///
    /// Irreducibility is NOT checked, because the check is expensive relative to field construction.
    /// Use gf2_is_irreducible if the polynomial comes from an untrusted source.
    GF2Field(size_t degree, const GF2Poly &irreducible_poly);

    /// The extension degree m of the field.
    inline size_t degree() const {
        return degree_;
    }
    /// The number of elements in the field is 2^degree().
    ///
    /// The reduction polynomial, including its x^degree() term.
    inline const GF2Poly &modulus() const {
        return modulus_;
    }

    /// The additive identity.
    GF2Poly zero() const;
    /// The multiplicative identity.
    GF2Poly one() const;
    /// The element x, i.e. the polynomial with only the degree 1 coefficient set.
    ///
    /// For degree 1 fields there is no element x (the only elements are 0 and 1), so this returns 1.
    GF2Poly x() const;

    /// Returns true if the value is a valid element of the field, i.e. has degree less than m.
    bool is_element(const GF2Poly &v) const;
    /// Reduces an arbitrary polynomial into the field.
    GF2Poly mod(const GF2Poly &v) const;

    /// Reduces `v` in place. Requires degree(v) < 2 * degree().
    void ireduce(GF2Poly &v) const;

    /// Returns a + b, which over GF(2^m) is just XOR.
    inline GF2Poly add(const GF2Poly &a, const GF2Poly &b) const {
        return a ^ b;
    }
    /// Returns a * b.
    GF2Poly mul(const GF2Poly &a, const GF2Poly &b) const;
    /// Returns a * a, which is cheaper than mul(a, a) because squaring over GF(2) just spreads bits.
    GF2Poly square(const GF2Poly &a) const;
    /// Returns a^(2^k), i.e. the k-th power of the Frobenius endomorphism applied to a.
    GF2Poly frobenius(const GF2Poly &a, size_t k) const;
    /// Returns a^exponent.
    GF2Poly pow(const GF2Poly &a, uint64_t exponent) const;
    /// Returns a^exponent, for exponents too large to fit in a uint64_t.
    GF2Poly pow(const GF2Poly &a, const FixedWidthInt &exponent) const;
    /// Returns the multiplicative inverse of a, or zero when a is zero.
    ///
    /// Uses the extended Euclidean algorithm, which costs O(m^2) bit operations. This is far cheaper
    /// than the Fermat's-little-theorem approach of raising to the power 2^m - 2, which needs about
    /// 2m field multiplications.
    GF2Poly invert(const GF2Poly &a) const;
    /// Returns a / b, or zero when b is zero.
    GF2Poly div(const GF2Poly &a, const GF2Poly &b) const;

    /// Returns the matrix of the linear map a -> a^(2^k).
    ///
    /// Squaring is GF(2)-linear (the Frobenius endomorphism), so it can be realized by a circuit of
    /// CNOT gates synthesized from this matrix.
    GF2Matrix frobenius_matrix(size_t k) const;

    /// Returns the matrix of the linear map a -> constant * a. Requires constant != 0.
    GF2Matrix constant_mul_matrix(const GF2Poly &constant) const;

    /// Returns the matrix of a linear map, by applying `func` to each basis element.
    ///
    /// The caller is responsible for `func` actually being GF(2)-linear.
    template <typename F>
    GF2Matrix linear_map_matrix(F &&func) const {
        GF2Matrix result(degree_, degree_);
        for (size_t j = 0; j < degree_; j++) {
            GF2Poly image = func(GF2Poly::monomial(j));
            for (size_t i = 0; i < degree_; i++) {
                if (image.bit(i)) {
                    result.set(i, j, true);
                }
            }
        }
        return result;
    }

   private:
    void init();

    size_t degree_;
    GF2Poly modulus_;
    /// Exponents of the non-leading terms of the modulus, descending. Used for fast reduction.
    std::vector<uint32_t> mod_terms_;
    /// True when the modulus has few enough terms for the sparse reduction to beat the dense one.
    bool sparse_reduction_ = false;
};

/// Returns a low weight irreducible polynomial of the given degree, including its x^degree term.
///
/// Prefers trinomials (x^m + x^a + 1), falling back to pentanomials (x^m + x^a + x^b + x^c + 1) for
/// the degrees where no irreducible trinomial exists. Results are cached, since the search is much
/// more expensive than the arithmetic that uses it.
///
/// Requires 1 <= degree <= GF2_MAX_DEGREE.
GF2Poly gf2_default_irreducible_poly(size_t degree);

/// Returns true if the polynomial is irreducible over GF(2).
///
/// Uses Ben-Or's algorithm: a degree m polynomial is irreducible exactly when it shares no factor
/// with x^(2^d) - x for every d up to m / 2. Testing the small degrees first means reducible inputs,
/// which almost always have a small factor, are rejected quickly.
bool gf2_is_irreducible(const GF2Poly &poly);

}  // namespace kickmix

#endif
