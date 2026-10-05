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
/// The reduction polynomial is either supplied by the caller (any irreducible polynomial of degree
/// m) or defaults to gf2_default_irreducible_poly(m), which has low weight. Low weight moduli let
/// `ireduce` clear each high coefficient with a handful of bit flips; dense moduli fall back to a
/// word-at-a-time reduction, and make some circuit constructions (e.g. gen_gf_phase_by_product)
/// proportionally more expensive.
struct GF2Field {
    /// Creates GF(2^m) using gf2_default_irreducible_poly(degree).
    explicit GF2Field(size_t degree);

    /// Creates GF(2^m) using an explicit reduction polynomial.
    ///
    /// The polynomial must be irreducible and have degree `degree`. As a convenience the
    /// x^degree term may be omitted from `irreducible_poly`, in which case it is implied.
    ///
    /// Irreducibility is checked by default (`check_irreducible = true`) and throws
    /// `std::invalid_argument` if the polynomial is reducible.
    GF2Field(size_t degree, const GF2Poly &irreducible_poly, bool check_irreducible = true);

    /// Creates GF(2^m) using an explicit reduction polynomial and an explicit primitive element
    /// (multiplicative generator of GF(2^m)* of order 2^m - 1).
    ///
    /// Throws `std::invalid_argument` if `irreducible_poly` is reducible (when `check_irreducible`
    /// is true) or if `primitive_element` is not a primitive element of the field.
    GF2Field(
        size_t degree,
        const GF2Poly &irreducible_poly,
        const GF2Poly &primitive_element,
        bool check_irreducible = true);

    /// Creates GF(2^m) using an explicit reduction polynomial parsed from an algebraic
    /// ("x^100 + x^10 + x^2 + 1"), hex ("0x11B"), or binary ("0b100011011") string.
    ///
    /// Same semantics as the GF2Poly overload: the x^degree term may be omitted, and
    /// irreducibility is checked unless `check_irreducible` is false.
    GF2Field(size_t degree, std::string_view irreducible_poly_str, bool check_irreducible = true);

    /// Creates GF(2^m) by parsing an algebraic ("x^100 + x^10 + x^2 + 1"), hex ("0x11B"), or
    /// binary ("0b100011011") irreducible polynomial string. The field degree m is inferred from
    /// the degree of the parsed polynomial, so the leading term must be written out.
    explicit GF2Field(std::string_view irreducible_poly_str, bool check_irreducible = true);

    /// The extension degree m of the field. The field has 2^m elements.
    inline size_t degree() const {
        return degree_;
    }
    /// The reduction polynomial, including its x^degree() term.
    inline const GF2Poly &modulus() const {
        return modulus_;
    }

    /// Returns the primitive element (multiplicative generator of GF(2^m)* of order 2^m - 1)
    /// associated with this field.
    ///
    /// If no primitive element was explicitly supplied at construction time, this returns (and
    /// caches) the smallest primitive element in integer order (1 for m = 1, or the smallest
    /// polynomial in {x, x + 1, x^2, ...} of multiplicative order 2^m - 1 for m >= 2).
    GF2Poly primitive_element() const;

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

    /// True when a primitive element was supplied at construction time and it differs from the
    /// default (smallest) one. Fields for which this is false all share the default primitive
    /// element for their modulus, which lets equality and hashing skip the default search.
    inline bool has_custom_primitive_element() const {
        return has_custom_primitive_element_;
    }

    /// Fields are equal when they have the same degree, modulus and primitive element.
    ///
    /// The primitive element search is only performed when at least one side has a custom
    /// primitive element; otherwise both sides necessarily share the default one.
    inline bool operator==(const GF2Field &other) const {
        if (degree_ != other.degree_ || modulus_ != other.modulus_) {
            return false;
        }
        if (!has_custom_primitive_element_ && !other.has_custom_primitive_element_) {
            return true;
        }
        return primitive_element() == other.primitive_element();
    }
    inline bool operator!=(const GF2Field &other) const {
        return !(*this == other);
    }

   private:
    void init();

    size_t degree_;
    GF2Poly modulus_;
    /// Exponents of the non-leading terms of the modulus, descending. Used for fast reduction.
    std::vector<uint32_t> mod_terms_;
    /// True when the modulus has few enough terms for the sparse reduction to beat the dense one.
    bool sparse_reduction_ = false;
    /// Explicitly supplied primitive element, or zero when using the default (smallest) one.
    GF2Poly primitive_element_;
    bool has_custom_primitive_element_ = false;
};

/// Returns a low weight irreducible polynomial of the given degree, including its x^degree term.
///
/// For degrees up to 12 this is a fixed table of conventional low-weight irreducible polynomials
/// (e.g. 0x13 for GF(2^4) and 0x11D for GF(2^8)); note that these are not all primitive (for
/// m = 12, x^12 + x^3 + 1 is irreducible but x has order 1365, so use GF2Field::primitive_element()
/// rather than assuming x generates the multiplicative group). For larger degrees it is found by
/// search: the trinomial x^m + x^a + 1 with the smallest a, or if no irreducible trinomial exists,
/// the first irreducible pentanomial x^m + x^a + x^b + x^c + 1 in (a, b, c) lexicographic order.
/// Searched results are cached, since the search is much more expensive than the arithmetic that
/// uses it.
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
