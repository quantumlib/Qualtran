#ifndef _KICKMIX_UTIL_GF2_FIELD_H
#define _KICKMIX_UTIL_GF2_FIELD_H

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "kickmix/util/fixed_width_int.h"
#include "kickmix/util/gf2_matrix.h"
#include "kickmix/util/gf2_poly.h"

namespace kickmix {

/// Unchecked arithmetic in GF(2)[x] / (modulus) on already reduced polynomials (degree < `degree`).
///
/// This is the arithmetic core of GF2Field, which adds validation and element bookkeeping on top.
/// It is also used directly on candidate polynomials (by gf2_is_irreducible and the default
/// polynomial search) where constructing a validated GF2Field would be circular.
struct GF2PolyMod {
    size_t degree;
    GF2Poly modulus;
    /// Exponents of the non-leading terms of the modulus, descending. Used for fast reduction.
    std::vector<uint32_t> mod_terms;
    /// True when the modulus has few enough terms for the sparse reduction to beat the dense one.
    bool sparse_reduction;

    GF2PolyMod(size_t degree, const GF2Poly &modulus);

    /// Reduces `v` in place. Requires degree(v) < 2 * degree.
    void ireduce(GF2Poly &v) const;
    GF2Poly mul(const GF2Poly &a, const GF2Poly &b) const;
    GF2Poly square(const GF2Poly &a) const;
    GF2Poly pow(const GF2Poly &a, const FixedWidthInt &exponent) const;
    /// Ben-Or's irreducibility test on `modulus`. Requires degree >= 2 and modulus.bit(0).
    bool is_irreducible() const;
    /// Returns true if the non-zero reduced element `a` has multiplicative order 2^degree - 1
    /// (assuming `modulus` is irreducible).
    bool is_primitive_element(const GF2Poly &a) const;
};

/// Arithmetic in the binary extension field GF(2^m), for 1 <= m <= GF2_MAX_DEGREE.
///
/// Elements are polynomials of degree less than m over GF(2), represented as GF2Poly values whose
/// bit k is the coefficient of x^k. Addition is XOR, and multiplication is polynomial multiplication
/// reduced modulo an irreducible polynomial of degree m.
///
/// The reduction polynomial is either supplied by the caller (any irreducible polynomial of degree
/// m) or defaults to gf2_default_irreducible_poly(m), which is a low weight primitive polynomial
/// (so x is a primitive element of the default field). Low weight moduli let
/// `ireduce` clear each high coefficient with a handful of bit flips; dense moduli fall back to a
/// word-at-a-time reduction, and make some circuit constructions (e.g. gen_gf_phase_by_product)
/// proportionally more expensive.
struct GF2Field {
    /// Creates GF(2^m) using gf2_default_irreducible_poly(degree).
    explicit GF2Field(size_t degree);

    /// Creates GF(2^m) from an irreducible polynomial, inferring the field degree m from the
    /// degree of `irreducible_poly` (so the x^m leading term must be present).
    explicit GF2Field(const GF2Poly &irreducible_poly, std::optional<GF2Poly> primitive_element = std::nullopt);

    /// Creates GF(2^m) using an explicit reduction polynomial and, optionally, an explicit
    /// primitive element (multiplicative generator of GF(2^m)* of order 2^m - 1).
    ///
    /// The polynomial must be irreducible and have degree `degree`. As a convenience the
    /// x^degree term may be omitted from `irreducible_poly`, in which case it is implied.
    ///
    /// Throws `std::invalid_argument` if the polynomial is reducible or has the wrong degree, or
    /// if `primitive_element` is given but is not a primitive element of the field.
    GF2Field(size_t degree, const GF2Poly &irreducible_poly, std::optional<GF2Poly> primitive_element = std::nullopt);

    /// Creates GF(2^m) using an explicit reduction polynomial parsed from an algebraic
    /// ("x^100 + x^10 + x^2 + 1"), hex ("0x11B"), or binary ("0b100011011") string.
    ///
    /// Same semantics as the GF2Poly overload: the x^degree term may be omitted, and
    /// irreducibility is validated.
    GF2Field(size_t degree, std::string_view irreducible_poly_str);

    /// Creates GF(2^m) by parsing an algebraic ("x^100 + x^10 + x^2 + 1"), hex ("0x11B"), or
    /// binary ("0b100011011") irreducible polynomial string. The field degree m is inferred from
    /// the degree of the parsed polynomial, so the leading term must be written out.
    explicit GF2Field(std::string_view irreducible_poly_str);

    /// The extension degree m of the field. The field has 2^m elements.
    inline size_t degree() const {
        return mod_.degree;
    }
    /// The reduction polynomial, including its x^degree() term.
    inline const GF2Poly &modulus() const {
        return mod_.modulus;
    }

    /// Returns the primitive element (multiplicative generator of GF(2^m)* of order 2^m - 1)
    /// associated with this field.
    ///
    /// If no primitive element was explicitly supplied at construction time, this is the smallest
    /// primitive element in integer order (1 for m = 1, x for every default field with m >= 2, or
    /// the smallest polynomial in {x, x + 1, x^2, ...} of multiplicative order 2^m - 1 for custom
    /// moduli).
    inline const GF2Poly &primitive_element() const {
        return primitive_element_;
    }

    /// Returns true if `a` is a primitive element of this field, i.e. a non-zero element of
    /// GF(2^m) whose multiplicative order is 2^m - 1.
    bool is_primitive_element(const GF2Poly &a) const;

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
        size_t m = degree();
        GF2Matrix result(m, m);
        for (size_t j = 0; j < m; j++) {
            GF2Poly image = func(GF2Poly::monomial(j));
            for (size_t i = 0; i < m; i++) {
                if (image.bit(i)) {
                    result.set(i, j, true);
                }
            }
        }
        return result;
    }

    /// Fields are equal when they have the same degree, modulus and primitive element.
    inline bool operator==(const GF2Field &other) const {
        return degree() == other.degree() && modulus() == other.modulus() &&
               primitive_element_ == other.primitive_element_;
    }
    inline bool operator!=(const GF2Field &other) const {
        return !(*this == other);
    }

   private:
    GF2PolyMod mod_;
    /// Primitive element (multiplicative generator of GF(2^m)* of order 2^m - 1).
    GF2Poly primitive_element_;
};

/// Returns a low weight primitive polynomial of the given degree, including its x^degree term.
///
/// The result is the trinomial x^m + x^a + 1 with the smallest a that is primitive (irreducible
/// with x generating the multiplicative group), or if there is none, the first primitive
/// pentanomial x^m + x^a + x^b + x^c + 1 in (a, b, c) lexicographic order. Consequently x is a
/// primitive element of GF2Field(m) for every m. Results are cached, since the search is much more
/// expensive than the arithmetic that uses it.
///
/// Requires 1 <= degree <= GF2_MAX_DEGREE.
GF2Poly gf2_default_irreducible_poly(size_t degree);

/// Returns true if the polynomial is irreducible over GF(2).
///
/// Uses Ben-Or's algorithm: a degree m polynomial is irreducible exactly when it shares no factor
/// with x^(2^d) - x for every d up to m / 2. Testing the small degrees first means reducible inputs,
/// which almost always have a small factor, are rejected quickly.
bool gf2_is_irreducible(const GF2Poly &poly);

/// Returns the distinct prime factors of 2^degree - 1, each as a 512-bit FixedWidthInt.
///
/// This is the factorization used by GF2Field::is_primitive_element: g is primitive iff
/// g^((2^degree - 1) / p) != 1 for every prime factor p. It is computed from the tabulated
/// factors in gf2_cyclotomic_factors.h and is exposed so that table can be verified by tests.
///
/// Requires 1 <= degree <= GF2_MAX_DEGREE.
std::vector<FixedWidthInt> gf2_mersenne_prime_factors(size_t degree);

}  // namespace kickmix

#endif
