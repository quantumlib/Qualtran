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

/// Default moduli for small degrees. These are conventional low-weight irreducible polynomials
/// (for example 0x11D is the GF(2^8) primitive polynomial used by Reed-Solomon codes, not the AES
/// polynomial 0x11B), pinned so that `GF2Field(m)` for small m names a familiar, stable field.
/// Every entry is a trinomial when an irreducible trinomial of degree m exists, and a pentanomial
/// otherwise (degree 8).
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
    0x1009  // x^12 + x^3 + 1
};
constexpr size_t NUM_SMALL_DEGREE_POLYS = sizeof(SMALL_DEGREE_POLYS) / sizeof(SMALL_DEGREE_POLYS[0]);

std::string format_term(size_t exp) {
    if (exp == 0) {
        return "1";
    }
    if (exp == 1) {
        return "x";
    }
    return "x^" + std::to_string(exp);
}

std::string format_poly(const GF2Poly &poly) {
    size_t deg = poly.degree();
    if (deg == SIZE_MAX) {
        return "0x0 (0)";
    }
    std::vector<size_t> terms;
    for (size_t k = deg + 1; k--;) {
        if (poly.bit(k)) {
            terms.push_back(k);
        }
    }
    std::string expr;
    constexpr size_t MAX_SHOWN_TERMS = 8;
    if (terms.size() <= MAX_SHOWN_TERMS) {
        for (size_t i = 0; i < terms.size(); i++) {
            if (i > 0) {
                expr += " + ";
            }
            expr += format_term(terms[i]);
        }
    } else {
        for (size_t i = 0; i < 4; i++) {
            if (i > 0) {
                expr += " + ";
            }
            expr += format_term(terms[i]);
        }
        expr += " + ... + " + format_term(terms[terms.size() - 2]) + " + " + format_term(terms.back());
    }
    return poly.str() + " (" + expr + ")";
}

/// Runs Ben-Or's test on `field.modulus()`. Returns the zero polynomial if the modulus is
/// irreducible. Otherwise returns g = gcd(x^(2^d) - x, modulus) for the smallest d at which g is
/// non-trivial, and stores d into `*factor_degree`. Requires `field.degree() >= 2`.
///
/// Because no smaller d succeeded, every irreducible factor of g has degree exactly d, and g is
/// squarefree (x^(2^d) - x is). Note that g can be composite, and can even be the whole modulus
/// (e.g. x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 is the product of both irreducible cubics).
GF2Poly ben_or_gcd(const GF2Field &field, size_t *factor_degree) {
    size_t m = field.degree();
    const GF2Poly &poly = field.modulus();
    GF2Poly monomial_x = GF2Poly::monomial(1);
    GF2Poly one = GF2Poly::from_u64(1);
    GF2Poly h = monomial_x;
    for (size_t i = 1; i <= m / 2; i++) {
        h = field.square(h);
        GF2Poly g = GF2Poly::gcd(h ^ monomial_x, poly);
        if (g != one) {
            if (factor_degree != nullptr) {
                *factor_degree = i;
            }
            return g;
        }
    }
    return GF2Poly();
}

uint64_t splitmix64(uint64_t &state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/// Returns one irreducible factor of `g`, given that `g` is squarefree and all of its irreducible
/// factors have degree `d`.
///
/// Uses the characteristic 2 Cantor-Zassenhaus split: for a in GF(2)[x]/(g), the trace
/// T(a) = a + a^2 + ... + a^(2^(d-1)) is 0 or 1 modulo each factor, so g = gcd(T(a), g) *
/// gcd(T(a) + 1, g), and a random a separates any two factors with probability 1/2. The random
/// choices come from a fixed seed, so error messages are deterministic.
GF2Poly split_equal_degree_factor(GF2Poly g, size_t d) {
    uint64_t rng_state = 0x243F6A8885A308D3ull;
    GF2Poly one = GF2Poly::from_u64(1);
    size_t attempts = 0;
    while (g.degree() > d) {
        size_t n = g.degree();
        GF2Field ring(n, g, /*check_irreducible=*/false);
        bool split = false;
        while (!split) {
            if (++attempts > 10000) {
                // Unreachable for valid inputs (each attempt fails with probability <= 1/2).
                return g;
            }
            GF2Poly a;
            for (size_t w = 0; w < (n + 63) / 64; w++) {
                a.words[w] = splitmix64(rng_state);
            }
            a.truncate(n);
            GF2Poly term = a;
            GF2Poly trace = a;
            for (size_t i = 1; i < d; i++) {
                term = ring.square(term);
                trace ^= term;
            }
            GF2Poly g0 = GF2Poly::gcd(trace, g);
            size_t d0 = g0.degree();
            if (d0 == SIZE_MAX || d0 == 0 || d0 == n) {
                continue;
            }
            GF2Poly g1 = GF2Poly::gcd(trace ^ one, g);
            // Recurse into the smaller half; it still has only degree d factors.
            g = g0.degree() <= g1.degree() ? g0 : g1;
            split = true;
        }
    }
    return g;
}

/// Returns an irreducible proper factor of `field.modulus()`, or the zero polynomial if the modulus
/// is irreducible. Requires `field.degree() >= 2`.
GF2Poly find_irreducible_factor(const GF2Field &field) {
    size_t d = 0;
    GF2Poly g = ben_or_gcd(field, &d);
    if (g.is_zero()) {
        return g;
    }
    return split_equal_degree_factor(g, d);
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

GF2Poly parse_field_modulus_str(std::string_view irreducible_poly_str) {
    GF2Poly poly = GF2Poly::from_str(irreducible_poly_str);
    size_t d = poly.degree();
    if (d == SIZE_MAX || d < 1 || d > GF2_MAX_DEGREE) {
        std::string deg_desc = d == SIZE_MAX ? "-infinity (zero polynomial)" : std::to_string(d);
        throw std::invalid_argument(
            "GF2Field: modulus \"" + std::string(irreducible_poly_str) + "\" has degree " + deg_desc +
            ", which must be between 1 and " + std::to_string(GF2_MAX_DEGREE) +
            ". For example, use GF2Field(\"x^4 + x + 1\") or GF2Field(8).");
    }
    return poly;
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

GF2Poly find_default_primitive_element(const GF2Field &field) {
    size_t m = field.degree();
    if (m == 1) {
        return field.one();
    }

    static std::mutex mu;
    static std::vector<std::vector<std::pair<GF2Poly, GF2Poly>>> cache(GF2_MAX_DEGREE + 1);
    {
        std::lock_guard<std::mutex> lock(mu);
        for (const auto &entry : cache[m]) {
            if (entry.first == field.modulus()) {
                return entry.second;
            }
        }
    }

    GF2Poly found;
    for (uint64_t cand = 2;; cand++) {
        GF2Poly g = GF2Poly::from_u64(cand);
        if (field.is_primitive_element(g)) {
            found = g;
            break;
        }
    }

    {
        std::lock_guard<std::mutex> lock(mu);
        for (const auto &entry : cache[m]) {
            if (entry.first == field.modulus()) {
                return entry.second;
            }
        }
        cache[m].emplace_back(field.modulus(), found);
    }
    return found;
}

}  // namespace

GF2Field::GF2Field(size_t degree) : degree_(degree), modulus_(gf2_default_irreducible_poly(degree)) {
    init();
}

GF2Field::GF2Field(size_t degree, std::string_view irreducible_poly_str, bool check_irreducible)
    : GF2Field(degree, GF2Poly::from_str(irreducible_poly_str), check_irreducible) {
}

GF2Field::GF2Field(std::string_view irreducible_poly_str, bool check_irreducible)
    : GF2Field([&]() {
          GF2Poly poly = parse_field_modulus_str(irreducible_poly_str);
          return GF2Field(poly.degree(), poly, check_irreducible);
      }()) {
}

GF2Field::GF2Field(size_t degree, const GF2Poly &irreducible_poly, bool check_irreducible)
    : degree_(degree), modulus_(irreducible_poly) {
    if (degree < 1 || degree > GF2_MAX_DEGREE) {
        throw std::invalid_argument(
            "GF2Field: degree must be between 1 and " + std::to_string(GF2_MAX_DEGREE) + ", got " +
            std::to_string(degree) + ". For example, use GF2Field(8) for GF(2^8).");
    }
    // Allow the leading term to be omitted, but reject any polynomial of the wrong degree.
    size_t d = modulus_.degree();
    if (d == SIZE_MAX || d < degree) {
        modulus_.set_bit(degree, true);
    } else if (d != degree) {
        throw std::invalid_argument(
            "GF2Field: modulus " + format_poly(irreducible_poly) + " has degree " + std::to_string(d) +
            ", expected degree " + std::to_string(degree) + " for GF(2^" + std::to_string(degree) + "). " +
            format_valid_field_example(degree, &irreducible_poly));
    }
    if (!modulus_.bit(0)) {
        std::string msg = "GF2Field: modulus " + format_poly(modulus_);
        if (modulus_ != irreducible_poly) {
            msg += " (from input " + irreducible_poly.str() + " with implied x^" + std::to_string(degree) + " bit)";
        }
        msg += " for GF(2^" + std::to_string(degree) +
               ") must have a non-zero constant term (bit 0 is 0, so it is divisible by x; "
               "not irreducible over GF(2)). " +
               format_valid_field_example(degree);
        throw std::invalid_argument(msg);
    }
    init();
    if (check_irreducible && degree_ >= 2) {
        GF2Poly factor = find_irreducible_factor(*this);
        if (!factor.is_zero()) {
            std::string msg = "GF2Field: modulus " + format_poly(modulus_);
            if (modulus_ != irreducible_poly) {
                msg +=
                    " (from input " + irreducible_poly.str() + " with implied x^" + std::to_string(degree_) + " bit)";
            }
            msg += " of degree " + std::to_string(degree_) + " is not irreducible over GF(2): divisible by " +
                   format_poly(factor) + ". " + format_valid_field_example(degree_, &irreducible_poly);
            throw std::invalid_argument(msg);
        }
    }
}

GF2Field::GF2Field(
    size_t degree, const GF2Poly &irreducible_poly, const GF2Poly &primitive_element, bool check_irreducible)
    : GF2Field(degree, irreducible_poly, check_irreducible) {
    if (primitive_element.is_zero() || !is_element(primitive_element)) {
        throw std::invalid_argument(
            "GF2Field: primitive_element " + format_poly(primitive_element) + " is not a non-zero element of GF(2^" +
            std::to_string(degree_) + "); expected 1 <= primitive_element < 2^" + std::to_string(degree_) +
            ". For example, " + format_poly(find_default_primitive_element(*this)) + " is a primitive element.");
    }
    if (!is_primitive_element(primitive_element)) {
        throw std::invalid_argument(
            "GF2Field: primitive_element " + format_poly(primitive_element) + " is not a primitive element of GF(2^" +
            std::to_string(degree_) + ") modulo " + format_poly(modulus_) +
            " (its multiplicative order is a proper divisor of 2^" + std::to_string(degree_) + " - 1). For example, " +
            format_poly(find_default_primitive_element(*this)) + " is a primitive element.");
    }
    primitive_element_ = primitive_element;
    has_custom_primitive_element_ = (primitive_element != find_default_primitive_element(*this));
}

void GF2Field::init() {
    if (degree_ < 1 || degree_ > GF2_MAX_DEGREE) {
        throw std::invalid_argument(
            "GF2Field: degree must be between 1 and " + std::to_string(GF2_MAX_DEGREE) + ", got " +
            std::to_string(degree_) + ". For example, use GF2Field(8) for GF(2^8).");
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

GF2Poly GF2Field::primitive_element() const {
    if (!primitive_element_.is_zero()) {
        return primitive_element_;
    }
    return find_default_primitive_element(*this);
}

bool GF2Field::is_primitive_element(const GF2Poly &a) const {
    if (a.is_zero() || !is_element(a)) {
        return false;
    }
    if (degree_ == 1) {
        return a == one();
    }
    if (a == one()) {
        return false;
    }
    const std::vector<FixedWidthInt> &exponents = maximal_proper_order_exponents(degree_);
    for (const FixedWidthInt &exp : exponents) {
        if (pow(a, exp) == one()) {
            return false;
        }
    }
    return true;
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

    GF2Field field(m, poly, /*check_irreducible=*/false);
    return ben_or_gcd(field, nullptr).is_zero();
}

namespace {

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
            std::to_string(degree) + ". For example, use GF2Field(8) for GF(2^8).");
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
