#include "kickmix/util/gf2_field.h"

#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include "kickmix/util/gf2_cyclotomic_factors.h"

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

namespace {

std::string format_poly(const GF2Poly &poly) {
    return poly.str() + " (" + poly.algebraic_str(8) + ")";
}

std::string format_valid_field_example(size_t degree, const GF2Poly *alt_poly = nullptr) {
    GF2Poly def = gf2_default_irreducible_poly(degree);
    std::string deg_s = std::to_string(degree);
    std::string msg = "For example, use GF2Field(" + deg_s + ") for the default modulus, or GF2Field(" + deg_s + ", " +
                      def.str() + ") for " + format_poly(def);
    if (alt_poly != nullptr) {
        size_t alt_deg = alt_poly->degree();
        if (alt_deg >= 1 && alt_deg <= GF2_MAX_DEGREE && alt_deg != degree && alt_poly->bit(0) &&
            gf2_is_irreducible(*alt_poly)) {
            msg += " (or use GF2Field(" + std::to_string(alt_deg) + ", " + alt_poly->str() + ") if you intended GF(2^" +
                   std::to_string(alt_deg) + "))";
        }
    }
    msg += ".";
    return msg;
}

size_t infer_modulus_degree(const GF2Poly &irreducible_poly) {
    size_t d = irreducible_poly.degree();
    if (d == SIZE_MAX || d < 1 || d > GF2_MAX_DEGREE) {
        std::string deg_desc = d == SIZE_MAX ? "-infinity (zero polynomial)" : std::to_string(d);
        throw std::invalid_argument(
            "GF2Field: modulus " + format_poly(irreducible_poly) + " has degree " + deg_desc +
            ", which must be between 1 and " + std::to_string(GF2_MAX_DEGREE) +
            ". For example, use GF2Field(\"x^4 + x + 1\") or GF2Field(8).");
    }
    return d;
}

FixedWidthInt mersenne_u512(size_t d) {
    FixedWidthInt res(512);
    for (size_t w = 0; w < d / 64; w++) {
        res.words[w] = ~uint64_t{0};
    }
    if (d % 64 != 0) {
        res.words[d / 64] = (uint64_t{1} << (d % 64)) - 1;
    }
    return res;
}

// Divides `num` by `den` in place (`num = num / den`) if `num % den == 0` and returns `true`.
// Otherwise leaves `num` unchanged and returns `false`.
bool try_exact_div_u512(FixedWidthInt &num, const FixedWidthInt &den) {
    size_t den_bits = den.num_bits_in_use();
    size_t rem_bits = num.num_bits_in_use();
    if (den_bits == 0 || rem_bits < den_bits) {
        return false;
    }
    FixedWidthInt rem = num;
    FixedWidthInt quot(512);
    for (size_t shift = rem_bits - den_bits + 1; shift--;) {
        bool borrow = false;
        rem.isub_shifted(den, shift, &borrow);
        if (borrow) {
            rem.iadd_shifted(den, shift);
        } else {
            quot.set_bit(shift, true);
        }
    }
    if (rem.non_zero()) {
        return false;
    }
    num = std::move(quot);
    return true;
}

}  // namespace

std::vector<FixedWidthInt> kickmix::gf2_mersenne_prime_factors(size_t degree) {
    if (degree < 1 || degree > GF2_MAX_DEGREE) {
        throw std::invalid_argument(
            "gf2_mersenne_prime_factors: degree must be between 1 and " + std::to_string(GF2_MAX_DEGREE) + ", got " +
            std::to_string(degree) + ".");
    }

    std::vector<size_t> divs;
    for (size_t d = 1; d <= degree; d++) {
        if (degree % d == 0) {
            divs.push_back(d);
        }
    }

    std::vector<FixedWidthInt> phi;
    phi.reserve(divs.size());
    for (size_t d : divs) {
        phi.push_back(mersenne_u512(d));
    }
    for (size_t i = 0; i < divs.size(); i++) {
        for (size_t j = i + 1; j < divs.size(); j++) {
            if (divs[j] % divs[i] == 0) {
                try_exact_div_u512(phi[j], phi[i]);
            }
        }
    }

    std::vector<FixedWidthInt> prime_factors;
    for (size_t idx = 0; idx < divs.size(); idx++) {
        size_t d = divs[idx];
        if (d == 1) {
            continue;
        }
        FixedWidthInt rem = phi[idx];
        // Strip any extrinsic prime factor p | d already found from a proper divisor of d.
        for (const FixedWidthInt &p : prime_factors) {
            if (p <= static_cast<uint64_t>(d)) {
                while (try_exact_div_u512(rem, p)) {
                }
            }
        }
        // Strip tabulated < 2^64 non-largest primitive prime factors of 2^d - 1.
        for (uint16_t k = CYCLOTOMIC_U64_OFFSETS[d - 1]; k < CYCLOTOMIC_U64_OFFSETS[d]; k++) {
            FixedWidthInt p(512, CYCLOTOMIC_U64_FACTORS[k]);
            if (try_exact_div_u512(rem, p)) {
                prime_factors.push_back(p);
                while (try_exact_div_u512(rem, p)) {
                }
            }
        }
        // Strip tabulated >= 2^64 non-largest primitive prime factors of 2^d - 1.
        for (const BigCyclotomicFactor &bf : CYCLOTOMIC_BIG_FACTORS) {
            if (bf.degree == d) {
                FixedWidthInt p(512);
                p.words[0] = bf.w0;
                p.words[1] = bf.w1;
                p.words[2] = bf.w2;
                if (try_exact_div_u512(rem, p)) {
                    prime_factors.push_back(p);
                    while (try_exact_div_u512(rem, p)) {
                    }
                }
            }
        }
        if (rem > uint64_t{1}) {
            prime_factors.push_back(rem);
        }
    }
    return prime_factors;
}

namespace {

// Returns the maximal proper divisors {(2^degree - 1) / p_i} of the multiplicative group order
// N = 2^degree - 1, where {p_i} are the distinct prime factors of N.
// An element g in GF(2^degree)* is primitive iff g^((2^degree - 1) / p_i) != 1 for all i.
const std::vector<FixedWidthInt> &maximal_proper_order_exponents(size_t degree) {
    static std::mutex mu;
    static std::vector<std::vector<FixedWidthInt>> cache(GF2_MAX_DEGREE + 1);
    static std::vector<bool> cached(GF2_MAX_DEGREE + 1, false);

    std::lock_guard<std::mutex> lock(mu);
    if (cached[degree]) {
        return cache[degree];
    }

    std::vector<FixedWidthInt> prime_factors = gf2_mersenne_prime_factors(degree);
    FixedWidthInt order = mersenne_u512(degree);
    std::vector<FixedWidthInt> exponents;
    exponents.reserve(prime_factors.size());
    for (const FixedWidthInt &p : prime_factors) {
        FixedWidthInt exp = order;
        try_exact_div_u512(exp, p);
        exponents.push_back(std::move(exp));
    }

    cache[degree] = std::move(exponents);
    cached[degree] = true;
    return cache[degree];
}

}  // namespace

GF2PolyMod::GF2PolyMod(size_t degree, const GF2Poly &modulus) : degree(degree), modulus(modulus) {
    for (size_t k = degree; k--;) {
        if (modulus.bit(k)) {
            mod_terms.push_back((uint32_t)k);
        }
    }
    // With few terms, clearing a high coefficient costs a handful of bit flips, which beats the
    // word-at-a-time dense reduction. With many terms the dense reduction wins.
    sparse_reduction = mod_terms.size() <= 16;
}

void GF2PolyMod::ireduce(GF2Poly &v) const {
    size_t deg = v.degree();
    if (deg == SIZE_MAX || deg < degree) {
        return;
    }
    if (!sparse_reduction) {
        v.imod(modulus);
        return;
    }
    // Repeatedly replace x^k (for k >= m) with x^(k-m) * (modulus - x^m). Every replacement
    // only touches coefficients below k, so a single downward sweep suffices.
    for (size_t k = deg + 1; k-- > degree;) {
        if (v.bit(k)) {
            v.xor_bit(k);
            size_t base = k - degree;
            for (uint32_t e : mod_terms) {
                v.xor_bit(base + e);
            }
        }
    }
}

GF2Poly GF2PolyMod::mul(const GF2Poly &a, const GF2Poly &b) const {
    GF2Poly result = GF2Poly::mul(a, b);
    ireduce(result);
    return result;
}

GF2Poly GF2PolyMod::square(const GF2Poly &a) const {
    // Squaring over GF(2) is linear: (sum a_k x^k)^2 = sum a_k x^(2k). So the square is just
    // the bits of `a` spread out with a zero between each pair, which is much cheaper than a
    // multiply.
    GF2Poly result;
    size_t deg = a.degree();
    if (deg == SIZE_MAX) {
        return result;
    }
    size_t words_used = deg / 64 + 1;
    for (size_t k = 0; k < words_used; k++) {
        uint64_t w = a.words[k];
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

GF2Poly GF2PolyMod::pow(const GF2Poly &a, const FixedWidthInt &exponent) const {
    GF2Poly result = GF2Poly::from_u64(1);
    GF2Poly base = a;
    size_t num_bits = exponent.num_bits_in_use();
    for (size_t k = 0; k < num_bits; k++) {
        if ((exponent.words[k / 64] >> (k & 63)) & 1) {
            result = mul(result, base);
        }
        base = square(base);
    }
    return result;
}

bool GF2PolyMod::is_irreducible() const {
    GF2Poly monomial_x = GF2Poly::monomial(1);
    GF2Poly one = GF2Poly::from_u64(1);
    GF2Poly h = monomial_x;
    for (size_t i = 1; i <= degree / 2; i++) {
        h = square(h);
        if (GF2Poly::gcd(h ^ monomial_x, modulus) != one) {
            return false;
        }
    }
    return true;
}

bool GF2PolyMod::is_primitive_element(const GF2Poly &a) const {
    GF2Poly one = GF2Poly::from_u64(1);
    if (degree == 1) {
        return a == one;
    }
    if (a == one) {
        return false;
    }
    for (const FixedWidthInt &exp : maximal_proper_order_exponents(degree)) {
        if (pow(a, exp) == one) {
            return false;
        }
    }
    return true;
}

namespace {

// Returns the smallest primitive element in integer order: 1 for m = 1, else the first of
// x, x + 1, x^2, ... with multiplicative order 2^m - 1.
GF2Poly find_default_primitive_element(const GF2PolyMod &mod) {
    if (mod.degree == 1) {
        return GF2Poly::from_u64(1);
    }
    for (uint64_t cand = 2;; cand++) {
        GF2Poly g = GF2Poly::from_u64(cand);
        if (mod.is_primitive_element(g)) {
            return g;
        }
    }
}

// Validates the degree and the shape of a user supplied modulus (implying the leading term if it
// was omitted), returning the full modulus. Irreducibility is checked separately by the caller.
GF2Poly checked_modulus(size_t degree, const GF2Poly &irreducible_poly) {
    if (degree < 1 || degree > GF2_MAX_DEGREE) {
        throw std::invalid_argument(
            "GF2Field: degree must be between 1 and " + std::to_string(GF2_MAX_DEGREE) + ", got " +
            std::to_string(degree) + ". For example, use GF2Field(8) for GF(2^8).");
    }
    GF2Poly modulus = irreducible_poly;
    size_t d = modulus.degree();
    if (d == SIZE_MAX || d < degree) {
        modulus.set_bit(degree, true);
    } else if (d != degree) {
        throw std::invalid_argument(
            "GF2Field: modulus " + format_poly(irreducible_poly) + " has degree " + std::to_string(d) +
            ", expected degree " + std::to_string(degree) + " for GF(2^" + std::to_string(degree) + "). " +
            format_valid_field_example(degree, &irreducible_poly));
    }
    if (!modulus.bit(0)) {
        std::string msg = "GF2Field: modulus " + format_poly(modulus);
        if (modulus != irreducible_poly) {
            msg += " (from input " + irreducible_poly.str() + " with implied x^" + std::to_string(degree) + " bit)";
        }
        msg += " for GF(2^" + std::to_string(degree) +
               ") must have a non-zero constant term (bit 0 is 0, so it is divisible by x; "
               "not irreducible over GF(2)). " +
               format_valid_field_example(degree);
        throw std::invalid_argument(msg);
    }
    return modulus;
}

}  // namespace

GF2Field::GF2Field(size_t degree) : mod_(degree, gf2_default_irreducible_poly(degree)), primitive_element_(x()) {
}

GF2Field::GF2Field(const GF2Poly &irreducible_poly, std::optional<GF2Poly> primitive_element)
    : GF2Field(infer_modulus_degree(irreducible_poly), irreducible_poly, std::move(primitive_element)) {
}

GF2Field::GF2Field(size_t degree, std::string_view irreducible_poly_str)
    : GF2Field(degree, GF2Poly::from_str(irreducible_poly_str)) {
}

GF2Field::GF2Field(std::string_view irreducible_poly_str) : GF2Field(GF2Poly::from_str(irreducible_poly_str)) {
}

GF2Field::GF2Field(size_t degree, const GF2Poly &irreducible_poly, std::optional<GF2Poly> primitive_element)
    : mod_(degree, checked_modulus(degree, irreducible_poly)) {
    std::string deg_s = std::to_string(degree);
    if (degree >= 2 && !mod_.is_irreducible()) {
        std::string msg = "GF2Field: modulus " + format_poly(modulus());
        if (modulus() != irreducible_poly) {
            msg += " (from input " + irreducible_poly.str() + " with implied x^" + deg_s + " bit)";
        }
        msg += " of degree " + deg_s + " is not irreducible over GF(2). " +
               format_valid_field_example(degree, &irreducible_poly);
        throw std::invalid_argument(msg);
    }
    if (!primitive_element.has_value()) {
        primitive_element_ = find_default_primitive_element(mod_);
        return;
    }
    if (primitive_element->is_zero() || !is_element(*primitive_element)) {
        throw std::invalid_argument(
            "GF2Field: primitive_element " + format_poly(*primitive_element) + " is not a non-zero element of GF(2^" +
            deg_s + "); expected 1 <= primitive_element < 2^" + deg_s + ". For example, " +
            format_poly(find_default_primitive_element(mod_)) + " is a primitive element.");
    }
    if (!mod_.is_primitive_element(*primitive_element)) {
        throw std::invalid_argument(
            "GF2Field: primitive_element " + format_poly(*primitive_element) + " is not a primitive element of GF(2^" +
            deg_s + ") modulo " + format_poly(modulus()) + " (its multiplicative order is a proper divisor of 2^" +
            deg_s + " - 1). For example, " + format_poly(find_default_primitive_element(mod_)) +
            " is a primitive element.");
    }
    primitive_element_ = *primitive_element;
}

bool GF2Field::is_primitive_element(const GF2Poly &a) const {
    return !a.is_zero() && is_element(a) && mod_.is_primitive_element(a);
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
    return d == SIZE_MAX || d < degree();
}

GF2Poly GF2Field::mod(const GF2Poly &v) const {
    GF2Poly result = v;
    result.imod(modulus());
    return result;
}

void GF2Field::ireduce(GF2Poly &v) const {
    mod_.ireduce(v);
}

GF2Poly GF2Field::mul(const GF2Poly &a, const GF2Poly &b) const {
    return mod_.mul(is_element(a) ? a : mod(a), is_element(b) ? b : mod(b));
}

GF2Poly GF2Field::square(const GF2Poly &a) const {
    return mod_.square(is_element(a) ? a : mod(a));
}

GF2Poly GF2Field::frobenius(const GF2Poly &a, size_t k) const {
    GF2Poly result = mod(a);
    k %= degree();
    for (size_t i = 0; i < k; i++) {
        result = mod_.square(result);
    }
    return result;
}

GF2Poly GF2Field::pow(const GF2Poly &a, uint64_t exponent) const {
    return mod_.pow(mod(a), FixedWidthInt(64, exponent));
}

GF2Poly GF2Field::pow(const GF2Poly &a, const FixedWidthInt &exponent) const {
    return mod_.pow(mod(a), exponent);
}

GF2Poly GF2Field::invert(const GF2Poly &a) const {
    return GF2Poly::invert_mod(mod(a), modulus());
}

GF2Poly GF2Field::div(const GF2Poly &a, const GF2Poly &b) const {
    return mul(a, invert(b));
}

GF2Matrix GF2Field::frobenius_matrix(size_t k) const {
    size_t m = degree();
    k %= m;

    // Column j is the image of the basis element x^j, which is (x^j)^(2^k) = (x^(2^k))^j. So the
    // columns are the successive powers of w = x^(2^k): k squarings plus m multiplications, rather
    // than a squaring per column or a matrix power.
    GF2Poly w = x();
    for (size_t i = 0; i < k; i++) {
        w = square(w);
    }

    GF2Matrix result(m, m);
    GF2Poly v = one();
    for (size_t j = 0; j < m; j++) {
        for (size_t i = 0; i < m; i++) {
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
    size_t m = degree();
    GF2Matrix result(m, m);
    GF2Poly v = c;
    for (size_t j = 0; j < m; j++) {
        for (size_t i = 0; i < m; i++) {
            if (v.bit(i)) {
                result.set(i, j, true);
            }
        }
        v <<= 1;
        if (v.bit(m)) {
            v.xor_bit(m);
            for (uint32_t e : mod_.mod_terms) {
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
    return GF2PolyMod(m, poly).is_irreducible();
}

namespace {

// Returns true if `poly` (of degree `degree >= 2` with non-zero constant term) is irreducible
// and x generates the multiplicative group of GF(2)[x] / (poly), i.e. `poly` is a primitive
// polynomial.
bool is_primitive_poly(size_t degree, const GF2Poly &poly) {
    GF2PolyMod mod(degree, poly);
    return mod.is_irreducible() && mod.is_primitive_element(GF2Poly::monomial(1));
}

GF2Poly search_primitive_poly(size_t degree) {
    GF2Poly base = GF2Poly::monomial(degree);
    base.set_bit(0, true);
    if (degree == 1) {
        return base;  // x + 1: GF(2)* is trivial, and x = 1 generates it.
    }

    // Trinomials x^m + x^a + 1, preferring the smallest a.
    for (size_t a = 1; a < degree; a++) {
        GF2Poly candidate = base;
        candidate.set_bit(a, true);
        if (is_primitive_poly(degree, candidate)) {
            return candidate;
        }
    }

    // Pentanomials x^m + x^a + x^b + x^c + 1, ordered by a then b then c. Every degree up to
    // GF2_MAX_DEGREE has a primitive trinomial or pentanomial (verified exhaustively in tests).
    for (size_t a = 3; a < degree; a++) {
        for (size_t b = 2; b < a; b++) {
            for (size_t c = 1; c < b; c++) {
                GF2Poly candidate = base;
                candidate.set_bit(a, true);
                candidate.set_bit(b, true);
                candidate.set_bit(c, true);
                if (is_primitive_poly(degree, candidate)) {
                    return candidate;
                }
            }
        }
    }

    throw std::runtime_error("Failed to find a primitive polynomial of degree " + std::to_string(degree));
}

}  // namespace

GF2Poly kickmix::gf2_default_irreducible_poly(size_t degree) {
    if (degree < 1 || degree > GF2_MAX_DEGREE) {
        throw std::invalid_argument(
            "gf2_default_irreducible_poly: degree must be between 1 and " + std::to_string(GF2_MAX_DEGREE) + ", got " +
            std::to_string(degree) + ". For example, use GF2Field(8) for GF(2^8).");
    }
    // The search is expensive relative to the arithmetic that uses it, so results are memoized.
    static std::mutex mu;
    static std::vector<GF2Poly> cache(GF2_MAX_DEGREE + 1);
    static std::vector<bool> cached(GF2_MAX_DEGREE + 1, false);

    std::lock_guard<std::mutex> lock(mu);
    if (!cached[degree]) {
        cache[degree] = search_primitive_poly(degree);
        cached[degree] = true;
    }
    return cache[degree];
}
