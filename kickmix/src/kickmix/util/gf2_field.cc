#include "kickmix/util/gf2_field.h"

#include <mutex>
#include <stdexcept>
#include <string>

using namespace kickmix;

/// Spreads the 32 low bits of a value out into the even positions of a 64 bit word.
static inline uint64_t spread_bits_32(uint64_t v) {
    v &= 0x00000000FFFFFFFFull;
    v = (v | (v << 16)) & 0x0000FFFF0000FFFFull;
    v = (v | (v << 8)) & 0x00FF00FF00FF00FFull;
    v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0Full;
    v = (v | (v << 2)) & 0x3333333333333333ull;
    v = (v | (v << 1)) & 0x5555555555555555ull;
    return v;
}

GF2Field::GF2Field(size_t degree) : degree_(degree), modulus_(gf2_default_irreducible_poly(degree)) {
    init();
}

GF2Field::GF2Field(size_t degree, const GF2Poly &irreducible_poly) : degree_(degree), modulus_(irreducible_poly) {
    if (degree < 1 || degree > GF2_MAX_DEGREE) {
        throw std::invalid_argument(
            "GF2Field: degree must be between 1 and " + std::to_string(GF2_MAX_DEGREE) + ", got " +
            std::to_string(degree));
    }
    // Allow the leading term to be omitted, but reject any polynomial of the wrong degree.
    size_t d = modulus_.degree();
    if (d == SIZE_MAX || d < degree) {
        modulus_.set_bit(degree, true);
    } else if (d != degree) {
        throw std::invalid_argument(
            "GF2Field: irreducible_poly has degree " + std::to_string(d) + ", expected " + std::to_string(degree));
    }
    init();
}

void GF2Field::init() {
    if (degree_ < 1 || degree_ > GF2_MAX_DEGREE) {
        throw std::invalid_argument(
            "GF2Field: degree must be between 1 and " + std::to_string(GF2_MAX_DEGREE) + ", got " +
            std::to_string(degree_));
    }
    if (!modulus_.bit(degree_)) {
        throw std::invalid_argument("GF2Field: modulus is missing its leading term");
    }
    if (!modulus_.bit(0)) {
        throw std::invalid_argument("GF2Field: modulus must have a non-zero constant term");
    }

    mod_terms_.clear();
    for (size_t k = degree_; k--;) {
        if (modulus_.bit(k)) {
            mod_terms_.push_back((uint32_t)k);
        }
    }
    // With few terms, clearing a high coefficient costs a handful of bit flips, which beats the
    // word-at-a-time dense reduction. With many terms the dense reduction wins.
    sparse_reduction_ = mod_terms_.size() <= 16;
}

GF2Poly GF2Field::zero() const {
    return GF2Poly();
}

GF2Poly GF2Field::one() const {
    return GF2Poly::from_u64(1);
}

GF2Poly GF2Field::x() const {
    return mod(GF2Poly::monomial(1));
}

bool GF2Field::is_element(const GF2Poly &v) const {
    size_t d = v.degree();
    return d == SIZE_MAX || d < degree_;
}

GF2Poly GF2Field::mod(const GF2Poly &v) const {
    GF2Poly result = v;
    result.imod(modulus_);
    return result;
}

void GF2Field::ireduce(GF2Poly &v) const {
    size_t deg = v.degree();
    if (deg == SIZE_MAX || deg < degree_) {
        return;
    }
    if (!sparse_reduction_) {
        v.imod(modulus_);
        return;
    }
    // Repeatedly replace x^k (for k >= m) with x^(k-m) * (modulus - x^m). Every replacement only
    // touches coefficients below k, so a single downward sweep suffices.
    for (size_t k = deg + 1; k-- > degree_;) {
        if (v.bit(k)) {
            v.xor_bit(k);
            size_t base = k - degree_;
            for (uint32_t e : mod_terms_) {
                v.xor_bit(base + e);
            }
        }
    }
}

GF2Poly GF2Field::mul(const GF2Poly &a, const GF2Poly &b) const {
    GF2Poly lhs = is_element(a) ? a : mod(a);
    GF2Poly rhs = is_element(b) ? b : mod(b);
    GF2Poly result = GF2Poly::mul(lhs, rhs);
    ireduce(result);
    return result;
}

GF2Poly GF2Field::square(const GF2Poly &a) const {
    // Squaring over GF(2) is linear: (sum a_k x^k)^2 = sum a_k x^(2k). So the square is just the
    // bits of `a` spread out with a zero between each pair, which is much cheaper than a multiply.
    GF2Poly v = is_element(a) ? a : mod(a);
    GF2Poly result;
    size_t deg = v.degree();
    if (deg == SIZE_MAX) {
        return result;
    }
    size_t words_used = deg / 64 + 1;
    for (size_t k = 0; k < words_used; k++) {
        uint64_t w = v.words[k];
        if (2 * k < GF2Poly::MAX_WORDS) {
            result.words[2 * k] = spread_bits_32(w);
        }
        if (2 * k + 1 < GF2Poly::MAX_WORDS) {
            result.words[2 * k + 1] = spread_bits_32(w >> 32);
        }
    }
    ireduce(result);
    return result;
}

GF2Poly GF2Field::frobenius(const GF2Poly &a, size_t k) const {
    GF2Poly result = mod(a);
    k %= degree_;
    for (size_t i = 0; i < k; i++) {
        result = square(result);
    }
    return result;
}

GF2Poly GF2Field::pow(const GF2Poly &a, uint64_t exponent) const {
    GF2Poly result = one();
    GF2Poly base = mod(a);
    while (exponent > 0) {
        if (exponent & 1) {
            result = mul(result, base);
        }
        base = square(base);
        exponent >>= 1;
    }
    return result;
}

GF2Poly GF2Field::pow(const GF2Poly &a, const FixedWidthInt &exponent) const {
    GF2Poly result = one();
    GF2Poly base = mod(a);
    size_t num_bits = exponent.num_bits_in_use();
    for (size_t k = 0; k < num_bits; k++) {
        if ((exponent.words[k / 64] >> (k & 63)) & 1) {
            result = mul(result, base);
        }
        base = square(base);
    }
    return result;
}

GF2Poly GF2Field::invert(const GF2Poly &a) const {
    return GF2Poly::invert_mod(mod(a), modulus_);
}

GF2Poly GF2Field::div(const GF2Poly &a, const GF2Poly &b) const {
    return mul(a, invert(b));
}

GF2Matrix GF2Field::frobenius_matrix(size_t k) const {
    k %= degree_;

    // Column j is the image of the basis element x^j, which is (x^j)^(2^k) = (x^(2^k))^j. So the
    // columns are the successive powers of w = x^(2^k): k squarings plus m multiplications, rather
    // than a squaring per column or a matrix power.
    GF2Poly w = x();
    for (size_t i = 0; i < k; i++) {
        w = square(w);
    }

    GF2Matrix result(degree_, degree_);
    GF2Poly v = one();
    for (size_t j = 0; j < degree_; j++) {
        for (size_t i = 0; i < degree_; i++) {
            if (v.bit(i)) {
                result.set(i, j, true);
            }
        }
        v = mul(v, w);
    }
    return result;
}

GF2Matrix GF2Field::constant_mul_matrix(const GF2Poly &constant) const {
    GF2Poly c = mod(constant);
    if (c.is_zero()) {
        throw std::invalid_argument("GF2Field::constant_mul_matrix: constant is zero (the map is not invertible)");
    }

    // Column j is constant * x^j, and each column is the previous one multiplied by x. Multiplying
    // by x is a shift plus at most one reduction step, so building the whole matrix is O(m^2 / 64).
    GF2Matrix result(degree_, degree_);
    GF2Poly v = c;
    for (size_t j = 0; j < degree_; j++) {
        for (size_t i = 0; i < degree_; i++) {
            if (v.bit(i)) {
                result.set(i, j, true);
            }
        }
        v <<= 1;
        if (v.bit(degree_)) {
            v.xor_bit(degree_);
            for (uint32_t e : mod_terms_) {
                v.xor_bit(e);
            }
        }
    }
    return result;
}

bool kickmix::gf2_is_irreducible(const GF2Poly &poly) {
    size_t m = poly.degree();
    if (m == SIZE_MAX || m == 0) {
        return false;  // The zero polynomial and the constant 1 are not irreducible.
    }
    if (m == 1) {
        return true;  // Both x and x + 1 are irreducible.
    }
    if (!poly.bit(0)) {
        return false;  // Divisible by x.
    }
    if (m > GF2_MAX_DEGREE) {
        throw std::invalid_argument("gf2_is_irreducible: degree exceeds GF2_MAX_DEGREE");
    }

    // Ben-Or's test. h tracks x^(2^i) mod poly; poly has an irreducible factor of degree dividing i
    // exactly when gcd(x^(2^i) - x, poly) is non-trivial. Over GF(2), subtraction is XOR.
    GF2Field field(m, poly);
    GF2Poly monomial_x = GF2Poly::monomial(1);
    GF2Poly one = GF2Poly::from_u64(1);
    GF2Poly h = monomial_x;
    for (size_t i = 1; i <= m / 2; i++) {
        h = field.square(h);
        if (GF2Poly::gcd(h ^ monomial_x, poly) != one) {
            return false;
        }
    }
    return true;
}

namespace {

/// Irreducible polynomials matching the ones the original implementation hardcoded, so that the
/// concrete field for a given small degree does not change.
constexpr uint64_t SMALL_DEGREE_POLYS[] = {
    0,      // unused
    0x3,    // x + 1
    0x7,    // x^2 + x + 1
    0xB,    // x^3 + x + 1
    0x13,   // x^4 + x + 1
    0x25,   // x^5 + x^2 + 1
    0x43,   // x^6 + x + 1
    0x89,   // x^7 + x^3 + 1
    0x11D,  // x^8 + x^4 + x^3 + x^2 + 1
    0x211,  // x^9 + x^4 + 1
    0x409,  // x^10 + x^3 + 1
    0x805,  // x^11 + x^2 + 1
    0x1053  // x^12 + x^6 + x^4 + x + 1
};
constexpr size_t NUM_SMALL_DEGREE_POLYS = sizeof(SMALL_DEGREE_POLYS) / sizeof(SMALL_DEGREE_POLYS[0]);

GF2Poly search_irreducible_poly(size_t degree) {
    GF2Poly base = GF2Poly::monomial(degree);
    base.set_bit(0, true);

    // Trinomials x^m + x^a + 1, preferring the smallest a.
    for (size_t a = 1; a < degree; a++) {
        GF2Poly candidate = base;
        candidate.set_bit(a, true);
        if (gf2_is_irreducible(candidate)) {
            return candidate;
        }
    }

    // Pentanomials x^m + x^a + x^b + x^c + 1, ordered by a then b then c. Every degree that has no
    // irreducible trinomial is known to have an irreducible pentanomial.
    for (size_t a = 3; a < degree; a++) {
        for (size_t b = 2; b < a; b++) {
            for (size_t c = 1; c < b; c++) {
                GF2Poly candidate = base;
                candidate.set_bit(a, true);
                candidate.set_bit(b, true);
                candidate.set_bit(c, true);
                if (gf2_is_irreducible(candidate)) {
                    return candidate;
                }
            }
        }
    }

    throw std::runtime_error("Failed to find an irreducible polynomial of degree " + std::to_string(degree));
}

}  // namespace

GF2Poly kickmix::gf2_default_irreducible_poly(size_t degree) {
    if (degree < 1 || degree > GF2_MAX_DEGREE) {
        throw std::invalid_argument(
            "gf2_default_irreducible_poly: degree must be between 1 and " + std::to_string(GF2_MAX_DEGREE) + ", got " +
            std::to_string(degree));
    }
    if (degree < NUM_SMALL_DEGREE_POLYS) {
        return GF2Poly::from_u64(SMALL_DEGREE_POLYS[degree]);
    }

    // The search is expensive relative to the arithmetic that uses it, so results are memoized.
    static std::mutex mu;
    static std::vector<GF2Poly> cache(GF2_MAX_DEGREE + 1);
    static std::vector<bool> cached(GF2_MAX_DEGREE + 1, false);

    std::lock_guard<std::mutex> lock(mu);
    if (!cached[degree]) {
        cache[degree] = search_irreducible_poly(degree);
        cached[degree] = true;
    }
    return cache[degree];
}
