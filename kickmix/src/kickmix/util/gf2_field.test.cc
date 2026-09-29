#include "gf2_field.h"

#include <algorithm>
#include <set>
#include <string>

#include "gtest/gtest.h"

#include "test.test.h"

using namespace kickmix;

static GF2Poly random_element(std::mt19937_64 &rng, const GF2Field &field) {
    GF2Poly result;
    for (size_t k = 0; k < field.degree(); k++) {
        result.set_bit(k, rng() & 1);
    }
    return result;
}

TEST(gf2_field, construction_rejects_bad_degrees) {
    ASSERT_THROW({ GF2Field(0); }, std::invalid_argument);
    ASSERT_THROW({ GF2Field(GF2_MAX_DEGREE + 1); }, std::invalid_argument);
    // A reduction polynomial of too high a degree is rejected. (A lower degree one is not: its
    // leading term is taken to have been omitted, which is covered by the next test.)
    ASSERT_THROW({ GF2Field(4, GF2Poly::from_u64(0x25)); }, std::invalid_argument);
    ASSERT_THROW({ GF2Field(2, GF2Poly::from_u64(0x11B)); }, std::invalid_argument);
    // Reducible polynomials are rejected by default in C++ as well.
    ASSERT_THROW({ GF2Field(4, GF2Poly::from_u64(0x15)); }, std::invalid_argument);
    ASSERT_THROW({ GF2Field(8, GF2Poly::from_u64(0x101)); }, std::invalid_argument);
}

TEST(gf2_field, construction_implies_leading_term) {
    // 0x1B without the x^8 term still means the AES polynomial when the degree says so.
    GF2Field a(8, GF2Poly::from_u64(0x11B));
    GF2Field b(8, GF2Poly::from_u64(0x1B));
    ASSERT_EQ(a.modulus(), b.modulus());
    ASSERT_EQ(a.modulus(), GF2Poly::from_u64(0x11B));
    ASSERT_EQ(a.degree(), 8);
}

TEST(gf2_field, basic_elements) {
    GF2Field field(8);
    ASSERT_TRUE(field.zero().is_zero());
    ASSERT_EQ(field.one(), GF2Poly::from_u64(1));
    ASSERT_EQ(field.x(), GF2Poly::from_u64(2));
    ASSERT_TRUE(field.is_element(GF2Poly::from_u64(0xFF)));
    ASSERT_FALSE(field.is_element(GF2Poly::from_u64(0x100)));

    // GF(2) is degenerate: there is no separate element x.
    GF2Field tiny(1);
    ASSERT_EQ(tiny.x(), tiny.one());
}

TEST(gf2_field, aes_field_known_values) {
    // The AES field, GF(2^8) mod x^8 + x^4 + x^3 + x + 1.
    GF2Field field(8, GF2Poly::from_u64(0x11B));

    // 0x57 * 0x83 = 0xC1 is the worked example from the AES specification.
    ASSERT_EQ(field.mul(GF2Poly::from_u64(0x57), GF2Poly::from_u64(0x83)), GF2Poly::from_u64(0xC1));
    // 0x57 * 0x13 = 0xFE, also from the specification.
    ASSERT_EQ(field.mul(GF2Poly::from_u64(0x57), GF2Poly::from_u64(0x13)), GF2Poly::from_u64(0xFE));
    // Multiplying by x is a shift with a conditional reduction.
    ASSERT_EQ(field.mul(GF2Poly::from_u64(0x57), GF2Poly::from_u64(0x02)), GF2Poly::from_u64(0xAE));
    ASSERT_EQ(field.mul(GF2Poly::from_u64(0xAE), GF2Poly::from_u64(0x02)), GF2Poly::from_u64(0x47));
    // The inverse of 0x53 is 0xCA (the AES S-box inversion step).
    ASSERT_EQ(field.invert(GF2Poly::from_u64(0x53)), GF2Poly::from_u64(0xCA));
}

TEST(gf2_field, gf4_multiplication_table) {
    // GF(4) mod x^2 + x + 1. Elements are 0, 1, x, x+1.
    GF2Field field(2);
    ASSERT_EQ(field.modulus(), GF2Poly::from_u64(0x7));
    GF2Poly zero = GF2Poly::from_u64(0);
    GF2Poly one = GF2Poly::from_u64(1);
    GF2Poly x = GF2Poly::from_u64(2);
    GF2Poly x1 = GF2Poly::from_u64(3);

    ASSERT_EQ(field.mul(x, x), x1);
    ASSERT_EQ(field.mul(x, x1), one);
    ASSERT_EQ(field.mul(x1, x1), x);
    ASSERT_EQ(field.mul(x, one), x);
    ASSERT_EQ(field.mul(x, zero), zero);
    ASSERT_EQ(field.invert(x), x1);
    ASSERT_EQ(field.invert(x1), x);
    ASSERT_EQ(field.invert(one), one);
    ASSERT_EQ(field.invert(zero), zero);
}

TEST(gf2_field, gf2_degenerate_field) {
    GF2Field field(1);
    GF2Poly zero = GF2Poly::from_u64(0);
    GF2Poly one = GF2Poly::from_u64(1);
    ASSERT_EQ(field.mul(one, one), one);
    ASSERT_EQ(field.mul(one, zero), zero);
    ASSERT_EQ(field.invert(one), one);
    ASSERT_EQ(field.square(one), one);
    ASSERT_EQ(field.frobenius(one, 5), one);
    ASSERT_EQ(field.frobenius_matrix(3), GF2Matrix::identity(1));
    ASSERT_EQ(field.constant_mul_matrix(one), GF2Matrix::identity(1));
}

TEST(gf2_field, exhaustive_small_field_axioms) {
    for (size_t m : std::vector<size_t>{1, 2, 3, 4, 5}) {
        GF2Field field(m);
        size_t n = size_t{1} << m;
        GF2Poly one = field.one();

        for (uint64_t ai = 0; ai < n; ai++) {
            GF2Poly a = GF2Poly::from_u64(ai);
            ASSERT_TRUE(field.is_element(a)) << m;
            // Inversion.
            GF2Poly inv = field.invert(a);
            if (ai == 0) {
                ASSERT_TRUE(inv.is_zero()) << m;
            } else {
                ASSERT_EQ(field.mul(a, inv), one) << m << " " << ai;
                ASSERT_EQ(field.div(a, a), one) << m << " " << ai;
            }
            // Squaring agrees with multiplication, and Frobenius has order m.
            ASSERT_EQ(field.square(a), field.mul(a, a)) << m << " " << ai;
            ASSERT_EQ(field.frobenius(a, m), a) << m << " " << ai;
            // Fermat: a^(2^m) == a.
            ASSERT_EQ(field.pow(a, uint64_t{1} << m), a) << m << " " << ai;

            for (uint64_t bi = 0; bi < n; bi++) {
                GF2Poly b = GF2Poly::from_u64(bi);
                GF2Poly prod = field.mul(a, b);
                ASSERT_TRUE(field.is_element(prod)) << m;
                // Commutativity.
                ASSERT_EQ(prod, field.mul(b, a)) << m;
                // Distributivity over addition.
                ASSERT_EQ(field.mul(a, field.add(b, one)), field.add(prod, a)) << m;
                // Freshman's dream: (a + b)^2 == a^2 + b^2.
                ASSERT_EQ(field.square(field.add(a, b)), field.add(field.square(a), field.square(b))) << m;
            }
        }
    }
}

TEST(gf2_field, associativity_fuzz) {
    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t m : std::vector<size_t>{7, 16, 63, 64, 65, 127, 128, 233, 256, 384, 512}) {
        GF2Field field(m);
        for (size_t trial = 0; trial < 10; trial++) {
            GF2Poly a = random_element(rng, field);
            GF2Poly b = random_element(rng, field);
            GF2Poly c = random_element(rng, field);
            ASSERT_EQ(field.mul(field.mul(a, b), c), field.mul(a, field.mul(b, c))) << m;
            ASSERT_EQ(field.mul(a, field.add(b, c)), field.add(field.mul(a, b), field.mul(a, c))) << m;
            ASSERT_TRUE(field.is_element(field.mul(a, b))) << m;
        }
    }
}

TEST(gf2_field, inversion_fuzz_including_512) {
    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t m : std::vector<size_t>{8, 31, 64, 65, 100, 163, 233, 256, 400, 512}) {
        GF2Field field(m);
        GF2Poly one = field.one();
        for (size_t trial = 0; trial < 10; trial++) {
            GF2Poly a = random_element(rng, field);
            if (a.is_zero()) {
                continue;
            }
            GF2Poly inv = field.invert(a);
            ASSERT_FALSE(inv.is_zero()) << m;
            ASSERT_TRUE(field.is_element(inv)) << m;
            ASSERT_EQ(field.mul(a, inv), one) << m;
            ASSERT_EQ(field.div(one, a), inv) << m;
        }
        ASSERT_TRUE(field.invert(field.zero()).is_zero()) << m;
    }
}

TEST(gf2_field, squaring_and_frobenius_fuzz) {
    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t m : std::vector<size_t>{2, 8, 63, 64, 65, 128, 257, 512}) {
        GF2Field field(m);
        for (size_t trial = 0; trial < 5; trial++) {
            GF2Poly a = random_element(rng, field);
            ASSERT_EQ(field.square(a), field.mul(a, a)) << m;
            // Frobenius has order m, so applying it m times is the identity.
            ASSERT_EQ(field.frobenius(a, m), a) << m;
            ASSERT_EQ(field.frobenius(a, 0), a) << m;
            ASSERT_EQ(field.frobenius(a, 1), field.square(a)) << m;
            // It is also additive.
            GF2Poly b = random_element(rng, field);
            size_t k = 1 + rng() % m;
            ASSERT_EQ(field.frobenius(field.add(a, b), k), field.add(field.frobenius(a, k), field.frobenius(b, k)))
                << m;
        }
    }
}

TEST(gf2_field, pow_matches_repeated_multiplication) {
    GF2Field field(8);
    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t trial = 0; trial < 20; trial++) {
        GF2Poly a = random_element(rng, field);
        uint64_t e = rng() % 50;
        GF2Poly expected = field.one();
        for (uint64_t k = 0; k < e; k++) {
            expected = field.mul(expected, a);
        }
        ASSERT_EQ(field.pow(a, e), expected) << e;
    }
    // The order of the multiplicative group divides 2^m - 1.
    GF2Poly g = GF2Poly::from_u64(0x53);
    ASSERT_EQ(field.pow(g, 255), field.one());
}

TEST(gf2_field, pow_with_big_exponent) {
    GF2Field field(64);
    auto rng = INDEPENDENT_TEST_RNG();
    GF2Poly a = random_element(rng, field);
    if (a.is_zero()) {
        a = field.one();
    }
    // Fermat's little theorem: a^(2^64 - 1) == 1.
    FixedWidthInt e(65);
    e.words[1] = 1;  // 2^64
    e -= (uint64_t)1;
    ASSERT_EQ(field.pow(a, e), field.one());
    // And the inverse is a^(2^64 - 2).
    e -= (uint64_t)1;
    ASSERT_EQ(field.pow(a, e), field.invert(a));
}

TEST(gf2_field, frobenius_matrix_matches_frobenius) {
    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t m : std::vector<size_t>{1, 2, 8, 31, 64, 65, 128, 512}) {
        GF2Field field(m);
        for (size_t k : std::vector<size_t>{0, 1, 2, m / 2, m - 1}) {
            GF2Matrix matrix = field.frobenius_matrix(k);
            ASSERT_EQ(matrix.rows, m);
            ASSERT_EQ(matrix.cols, m);
            for (size_t trial = 0; trial < 3; trial++) {
                GF2Poly a = random_element(rng, field);
                ASSERT_EQ(matrix * a, field.frobenius(a, k)) << m << " " << k;
            }
        }
        // Frobenius is a bijection, so its matrix is invertible.
        ASSERT_EQ(field.frobenius_matrix(1).rank(), m) << m;
        ASSERT_EQ(field.frobenius_matrix(0), GF2Matrix::identity(m)) << m;
        // Applying it m times is the identity.
        ASSERT_EQ(field.frobenius_matrix(m), GF2Matrix::identity(m)) << m;
    }
}

TEST(gf2_field, constant_mul_matrix_matches_mul) {
    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t m : std::vector<size_t>{1, 2, 8, 31, 64, 65, 128, 512}) {
        GF2Field field(m);
        for (size_t trial = 0; trial < 3; trial++) {
            GF2Poly c = random_element(rng, field);
            if (c.is_zero()) {
                c = field.one();
            }
            GF2Matrix matrix = field.constant_mul_matrix(c);
            ASSERT_EQ(matrix.rows, m);
            // Multiplication by a non-zero constant is invertible.
            ASSERT_EQ(matrix.rank(), m) << m;
            for (size_t k = 0; k < 3; k++) {
                GF2Poly a = random_element(rng, field);
                ASSERT_EQ(matrix * a, field.mul(c, a)) << m;
            }
        }
        ASSERT_EQ(field.constant_mul_matrix(field.one()), GF2Matrix::identity(m)) << m;
        ASSERT_THROW({ field.constant_mul_matrix(field.zero()); }, std::invalid_argument);
    }
}

TEST(gf2_field, linear_map_matrix) {
    GF2Field field(8);
    GF2Poly c = GF2Poly::from_u64(0x53);
    GF2Matrix matrix = field.linear_map_matrix([&](const GF2Poly &v) {
        return field.mul(c, v);
    });
    ASSERT_EQ(matrix, field.constant_mul_matrix(c));
}

TEST(gf2_field, is_irreducible_known_cases) {
    // Degenerate inputs.
    ASSERT_FALSE(gf2_is_irreducible(GF2Poly()));
    ASSERT_FALSE(gf2_is_irreducible(GF2Poly::from_u64(1)));
    // Degree 1: x and x + 1.
    ASSERT_TRUE(gf2_is_irreducible(GF2Poly::from_u64(0b10)));
    ASSERT_TRUE(gf2_is_irreducible(GF2Poly::from_u64(0b11)));
    // x^2 + x + 1 is the only irreducible quadratic.
    ASSERT_TRUE(gf2_is_irreducible(GF2Poly::from_u64(0b111)));
    ASSERT_FALSE(gf2_is_irreducible(GF2Poly::from_u64(0b100)));  // x^2
    ASSERT_FALSE(gf2_is_irreducible(GF2Poly::from_u64(0b101)));  // (x+1)^2
    ASSERT_FALSE(gf2_is_irreducible(GF2Poly::from_u64(0b110)));  // x(x+1)
    // The AES polynomial.
    ASSERT_TRUE(gf2_is_irreducible(GF2Poly::from_u64(0x11B)));
    // x^5 + x + 1 = (x^2 + x + 1)(x^3 + x^2 + 1) is reducible, unlike x^5 + x^2 + 1.
    ASSERT_FALSE(gf2_is_irreducible(GF2Poly::from_u64(0x23)));
    ASSERT_TRUE(gf2_is_irreducible(GF2Poly::from_u64(0x25)));
    // No trinomial of degree 8 is irreducible.
    for (size_t a = 1; a < 8; a++) {
        GF2Poly candidate = GF2Poly::monomial(8);
        candidate.set_bit(a, true);
        candidate.set_bit(0, true);
        ASSERT_FALSE(gf2_is_irreducible(candidate)) << a;
    }
}

TEST(gf2_field, default_irreducible_poly_small_degrees) {
    // The pinned small-degree defaults (conventional low-weight irreducible polynomials; not all
    // primitive, e.g. m = 12). Changing these would silently change which field GF2Field(m) refers
    // to, so they are locked down here.
    ASSERT_EQ(gf2_default_irreducible_poly(1), GF2Poly::from_u64(3));
    ASSERT_EQ(gf2_default_irreducible_poly(2), GF2Poly::from_u64(7));
    ASSERT_EQ(gf2_default_irreducible_poly(3), GF2Poly::from_u64(11));
    ASSERT_EQ(gf2_default_irreducible_poly(4), GF2Poly::from_u64(19));
    ASSERT_EQ(gf2_default_irreducible_poly(5), GF2Poly::from_u64(37));
    ASSERT_EQ(gf2_default_irreducible_poly(6), GF2Poly::from_u64(67));
    ASSERT_EQ(gf2_default_irreducible_poly(7), GF2Poly::from_u64(137));
    ASSERT_EQ(gf2_default_irreducible_poly(8), GF2Poly::from_u64(285));
    ASSERT_EQ(gf2_default_irreducible_poly(9), GF2Poly::from_u64(529));
    ASSERT_EQ(gf2_default_irreducible_poly(10), GF2Poly::from_u64(1033));
    ASSERT_EQ(gf2_default_irreducible_poly(11), GF2Poly::from_u64(2053));
    ASSERT_EQ(gf2_default_irreducible_poly(12), GF2Poly::from_u64(4105));

    ASSERT_THROW({ gf2_default_irreducible_poly(0); }, std::invalid_argument);
    ASSERT_THROW({ gf2_default_irreducible_poly(GF2_MAX_DEGREE + 1); }, std::invalid_argument);
}

TEST(gf2_field, default_irreducible_poly_is_irreducible) {
    // Every degree up to 64, plus a spread of larger ones including the maximum.
    std::vector<size_t> degrees;
    for (size_t m = 1; m <= 64; m++) {
        degrees.push_back(m);
    }
    for (size_t m : std::vector<size_t>{65, 100, 127, 128, 163, 233, 255, 256, 283, 384, 409, 511, 512}) {
        degrees.push_back(m);
    }

    for (size_t m : degrees) {
        GF2Poly poly = gf2_default_irreducible_poly(m);
        ASSERT_EQ(poly.degree(), m) << m;
        ASSERT_TRUE(poly.bit(0)) << m;
        // Low weight: a trinomial or a pentanomial.
        ASSERT_TRUE(poly.weight() == 3 || poly.weight() == 5 || m == 1) << m << " weight " << poly.weight();
        ASSERT_TRUE(gf2_is_irreducible(poly)) << m;
    }
}

TEST(gf2_field, default_irreducible_poly_is_cached) {
    // The second call must return the same polynomial, and come from the cache.
    GF2Poly first = gf2_default_irreducible_poly(300);
    GF2Poly second = gf2_default_irreducible_poly(300);
    ASSERT_EQ(first, second);
    ASSERT_TRUE(gf2_is_irreducible(first));
}

TEST(gf2_field, degree_512_end_to_end) {
    // The headline capability: a full 512 bit binary extension field.
    GF2Field field(512);
    ASSERT_EQ(field.degree(), 512);
    ASSERT_EQ(field.modulus().degree(), 512);

    auto rng = INDEPENDENT_TEST_RNG();
    GF2Poly a = random_element(rng, field);
    GF2Poly b = random_element(rng, field);
    if (a.is_zero()) {
        a = field.one();
    }
    if (b.is_zero()) {
        b = field.one();
    }

    GF2Poly prod = field.mul(a, b);
    ASSERT_TRUE(field.is_element(prod));
    ASSERT_EQ(field.div(prod, b), a);
    ASSERT_EQ(field.div(prod, a), b);
    ASSERT_EQ(field.mul(a, field.invert(a)), field.one());
    ASSERT_EQ(field.frobenius(a, 512), a);
    ASSERT_EQ(field.square(a), field.mul(a, a));
}

TEST(gf2_field, unreduced_inputs_and_zero_constant_term) {
    ASSERT_THROW({ GF2Field(1, GF2Poly::from_u64(0b10)); }, std::invalid_argument);
    ASSERT_THROW({ GF2Field(4, GF2Poly::from_u64(0b10010)); }, std::invalid_argument);

    GF2Field field(512);
    GF2Poly unreduced = GF2Poly::monomial(600) ^ GF2Poly::monomial(515) ^ GF2Poly::from_u64(0x12345);
    GF2Poly reduced = field.mod(unreduced);
    ASSERT_EQ(field.square(unreduced), field.square(reduced));
    ASSERT_EQ(field.mul(unreduced, unreduced), field.mul(reduced, reduced));
}

TEST(gf2_field, construction_from_string) {
    GF2Field aes_from_str("x^8 + x^4 + x^3 + x + 1");
    GF2Field aes_with_deg(8, "x^8 + x^4 + x^3 + x + 1");
    GF2Field aes_implied_top(8, "x^4 + x^3 + x + 1");
    ASSERT_EQ(aes_from_str.degree(), 8);
    ASSERT_EQ(aes_from_str.modulus(), GF2Poly::from_u64(0x11B));
    ASSERT_EQ(aes_with_deg.modulus(), GF2Poly::from_u64(0x11B));
    ASSERT_EQ(aes_implied_top.modulus(), GF2Poly::from_u64(0x11B));

    ASSERT_THROW({ GF2Field("x^4 + 1"); }, std::invalid_argument);
    ASSERT_THROW({ GF2Field("1"); }, std::invalid_argument);
    ASSERT_THROW({ GF2Field("0"); }, std::invalid_argument);
}

/// Returns the message of the std::invalid_argument thrown by constructing GF(2^degree) with
/// `modulus`, or fails the test if nothing is thrown.
static std::string reducible_modulus_error(size_t degree, const GF2Poly &modulus) {
    try {
        GF2Field field(degree, modulus);
    } catch (const std::invalid_argument &ex) {
        return ex.what();
    }
    ADD_FAILURE() << "GF2Field(" << degree << ", " << modulus << ") did not throw";
    return "";
}

/// Returns the nth irreducible polynomial of degree d, counting upward from x^d + 1.
static GF2Poly nth_irreducible(size_t d, size_t n) {
    GF2Poly candidate = GF2Poly::monomial(d);
    candidate.set_bit(0, true);
    while (true) {
        if (gf2_is_irreducible(candidate)) {
            if (n == 0) {
                return candidate;
            }
            n--;
        }
        // Step to the next odd polynomial of degree d, by incrementing the bits between 1 and d-1.
        for (size_t k = 1; k < d; k++) {
            candidate.xor_bit(k);
            if (candidate.bit(k)) {
                break;
            }
        }
    }
}

TEST(gf2_field, reducible_modulus_error_reports_an_irreducible_factor) {
    // x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = (x^3 + x + 1)(x^3 + x^2 + 1). Ben-Or's first non-trivial
    // gcd is the whole modulus here, which used to be reported as its own factor.
    std::string msg = reducible_modulus_error(6, GF2Poly::from_u64(0x7F));
    ASSERT_TRUE(
        msg.find("divisible by 0xB ") != std::string::npos || msg.find("divisible by 0xD ") != std::string::npos)
        << msg;

    // All three irreducible quartics multiplied together: the gcd is again the whole modulus, and
    // splitting it once still leaves a composite part.
    GF2Poly p = GF2Poly::mul(GF2Poly::mul(GF2Poly::from_u64(0x13), GF2Poly::from_u64(0x19)), GF2Poly::from_u64(0x1F));
    msg = reducible_modulus_error(12, p);
    ASSERT_TRUE(
        msg.find("divisible by 0x13 ") != std::string::npos || msg.find("divisible by 0x19 ") != std::string::npos ||
        msg.find("divisible by 0x1F ") != std::string::npos)
        << msg;

    // Products of two distinct equal degree irreducibles, up to the maximum field degree.
    for (size_t d : std::vector<size_t>{2, 3, 5, 8, 17, 31, 64, 65, 100, 128, 200, 256}) {
        GF2Poly a = nth_irreducible(d, 0);
        GF2Poly b = nth_irreducible(d, d == 2 ? 0 : 1);
        GF2Poly prod = GF2Poly::mul(a, b);
        msg = reducible_modulus_error(2 * d, prod);
        ASSERT_TRUE(
            msg.find("divisible by " + a.str() + " ") != std::string::npos ||
            msg.find("divisible by " + b.str() + " ") != std::string::npos)
            << "d=" << d << ": " << msg;
    }

    // A factor of smaller degree is still reported directly.
    msg = reducible_modulus_error(8, GF2Poly::from_u64(0x101));
    ASSERT_NE(msg.find("divisible by 0x3 "), std::string::npos) << msg;
}

TEST(gf2_field, construction_from_string_rejects_repeated_terms) {
    // Over GF(2) the repeated x^4 would cancel, turning this into a degree 1 modulus.
    ASSERT_THROW({ GF2Field("x^4 + x^4 + x + 1"); }, std::invalid_argument);
    ASSERT_THROW({ GF2Field(4, "x^4 + x + x + 1"); }, std::invalid_argument);
}

TEST(gf2_field, primitive_element) {
    GF2Field f1(1);
    ASSERT_EQ(f1.primitive_element(), GF2Poly::from_u64(1));
    ASSERT_TRUE(f1.is_primitive_element(GF2Poly::from_u64(1)));
    ASSERT_FALSE(f1.is_primitive_element(GF2Poly::from_u64(0)));

    for (size_t m = 2; m <= 11; m++) {
        GF2Field f(m);
        ASSERT_EQ(f.primitive_element(), GF2Poly::from_u64(2)) << m;
    }
    // Degree 12 default is the trinomial x^12 + x^3 + 1 (0x1009), whose smallest primitive element
    // is x + 1 (0x3). With the primitive pentanomial 0x1053, x (0x2) is primitive.
    GF2Field f12(12);
    ASSERT_EQ(f12.primitive_element(), GF2Poly::from_u64(3));
    ASSERT_FALSE(f12.is_primitive_element(GF2Poly::from_u64(2)));
    GF2Field f12_pent(12, GF2Poly::from_u64(0x1053));
    ASSERT_EQ(f12_pent.primitive_element(), GF2Poly::from_u64(2));

    // Verify that primitive_element() generates all 2^m - 1 non-zero elements for m <= 12.
    for (size_t m = 1; m <= 12; m++) {
        GF2Field f(m);
        GF2Poly g = f.primitive_element();
        size_t order = (size_t{1} << m) - 1;
        std::vector<bool> seen(size_t{1} << m, false);
        GF2Poly cur = f.one();
        for (size_t k = 0; k < order; k++) {
            uint64_t val = cur.words[0];
            ASSERT_GT(val, 0u);
            ASSERT_FALSE(seen[val]) << "m=" << m << " k=" << k;
            seen[val] = true;
            cur = f.mul(cur, g);
        }
        ASSERT_EQ(cur, f.one());
    }

    // AES polynomial 0x11B: x (0x2) has order 51, while x + 1 (0x3) is primitive.
    GF2Field aes(8, GF2Poly::from_u64(0x11B));
    ASSERT_FALSE(aes.is_primitive_element(GF2Poly::from_u64(2)));
    ASSERT_EQ(aes.primitive_element(), GF2Poly::from_u64(3));
    ASSERT_THROW({ GF2Field(8, GF2Poly::from_u64(0x11B), GF2Poly::from_u64(2)); }, std::invalid_argument);
    ASSERT_THROW({ GF2Field(8, GF2Poly::from_u64(0x11B), GF2Poly::from_u64(0)); }, std::invalid_argument);
    ASSERT_THROW({ GF2Field(8, GF2Poly::from_u64(0x11B), GF2Poly::from_u64(0x100)); }, std::invalid_argument);

    // Custom valid primitive element (0x9 = x^3 + 1 has order 15 in GF(2^4) mod 0x13).
    GF2Field f4_custom(4, GF2Poly::from_u64(0x13), GF2Poly::from_u64(0x9));
    ASSERT_EQ(f4_custom.primitive_element(), GF2Poly::from_u64(0x9));
    ASSERT_TRUE(f4_custom.has_custom_primitive_element());

    // Larger degrees up to 512.
    for (size_t m : std::vector<size_t>{14, 16, 31, 64, 100, 128, 256, 300, 512}) {
        GF2Field f(m);
        GF2Poly g = f.primitive_element();
        ASSERT_TRUE(f.is_primitive_element(g)) << m;
    }
}

// Deterministic Miller-Rabin for n < 2^64.
static bool is_prime_u64(uint64_t n) {
    if (n < 2) {
        return false;
    }
    for (uint64_t p : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull, 23ull, 29ull, 31ull, 37ull}) {
        if (n % p == 0) {
            return n == p;
        }
    }
    uint64_t d = n - 1;
    size_t s = 0;
    while ((d & 1) == 0) {
        d >>= 1;
        s++;
    }
    auto mulmod = [&](uint64_t a, uint64_t b) -> uint64_t {
        return static_cast<uint64_t>((static_cast<unsigned __int128>(a) * b) % n);
    };
    for (uint64_t a : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull, 23ull, 29ull, 31ull, 37ull}) {
        uint64_t x = 1;
        for (uint64_t base = a, e = d; e; e >>= 1, base = mulmod(base, base)) {
            if (e & 1) {
                x = mulmod(x, base);
            }
        }
        if (x == 1 || x == n - 1) {
            continue;
        }
        bool composite = true;
        for (size_t r = 1; r < s && composite; r++) {
            x = mulmod(x, x);
            composite = x != n - 1;
        }
        if (composite) {
            return false;
        }
    }
    return true;
}

// Miller-Rabin on 512-bit values with bases 2, 3, 5 and 7 (a strong probable prime test; the
// values checked here are published factors so any false positive would be a table bug, not a
// test flake).
static bool is_probable_prime_u512(const FixedWidthInt &n) {
    if (n.num_bits_in_use() <= 64) {
        return is_prime_u64(static_cast<uint64_t>(n));
    }
    FixedWidthInt n_minus_1 = n;
    n_minus_1.decrement();
    FixedWidthInt d = n_minus_1;
    size_t s = 0;
    while ((d.words[0] & 1) == 0) {
        d >>= uint64_t{1};
        s++;
    }
    auto mulmod = [&](const FixedWidthInt &a, const FixedWidthInt &b) {
        return (a * b) % n;
    };
    for (uint64_t a : {2ull, 3ull, 5ull, 7ull}) {
        FixedWidthInt base(512, a);
        FixedWidthInt x(512, uint64_t{1});
        for (size_t k = d.num_bits_in_use(); k--;) {
            x = mulmod(x, x);
            if ((d.words[k / 64] >> (k % 64)) & 1) {
                x = mulmod(x, base);
            }
        }
        if (x == uint64_t{1} || x == n_minus_1) {
            continue;
        }
        bool composite = true;
        for (size_t r = 1; r < s && composite; r++) {
            x = mulmod(x, x);
            composite = !(x == n_minus_1);
        }
        if (composite) {
            return false;
        }
    }
    return true;
}

TEST(gf2_field, mersenne_prime_factors_small_cases) {
    auto factors_u64 = [](size_t m) {
        std::vector<uint64_t> result;
        for (const FixedWidthInt &p : gf2_mersenne_prime_factors(m)) {
            result.push_back(static_cast<uint64_t>(p));
        }
        std::sort(result.begin(), result.end());
        return result;
    };
    ASSERT_EQ(factors_u64(1), (std::vector<uint64_t>{}));
    ASSERT_EQ(factors_u64(2), (std::vector<uint64_t>{3}));
    ASSERT_EQ(factors_u64(6), (std::vector<uint64_t>{3, 7}));
    ASSERT_EQ(factors_u64(11), (std::vector<uint64_t>{23, 89}));
    ASSERT_EQ(factors_u64(12), (std::vector<uint64_t>{3, 5, 7, 13}));
    ASSERT_EQ(factors_u64(31), (std::vector<uint64_t>{2147483647}));
    ASSERT_EQ(factors_u64(64), (std::vector<uint64_t>{3, 5, 17, 257, 641, 65537, 6700417}));
    ASSERT_THROW({ gf2_mersenne_prime_factors(0); }, std::invalid_argument);
    ASSERT_THROW({ gf2_mersenne_prime_factors(GF2_MAX_DEGREE + 1); }, std::invalid_argument);
}

// Validates the factor table in gf2_cyclotomic_factors.h: for every supported degree m, the
// factorization of 2^m - 1 derived from it must consist of distinct primes whose product (with
// multiplicity) is exactly 2^m - 1. If a tabulated entry were wrong or composite, or an entry were
// missing, the leftover cofactor would fail the primality check.
TEST(gf2_field, cyclotomic_factor_table_is_correct) {
    std::set<std::string> verified_primes;
    for (size_t m = 1; m <= GF2_MAX_DEGREE; m++) {
        FixedWidthInt order(512);
        order.set_bit(m, true);
        order.decrement();

        std::vector<FixedWidthInt> factors = gf2_mersenne_prime_factors(m);
        std::set<std::string> seen;
        FixedWidthInt product(512, uint64_t{1});
        for (const FixedWidthInt &p : factors) {
            ASSERT_TRUE(p > uint64_t{1}) << m;
            ASSERT_TRUE(seen.insert(p.hex()).second) << m << " repeated factor " << p.str();
            if (!verified_primes.count(p.hex())) {
                ASSERT_TRUE(is_probable_prime_u512(p)) << m << " composite factor " << p.str();
                verified_primes.insert(p.hex());
            }
            // Multiply in p as many times as it divides the order.
            while (true) {
                FixedWidthInt next = product * p;
                if ((order % next).non_zero()) {
                    break;
                }
                product *= p;
            }
        }
        ASSERT_TRUE(product == order) << m << " product of prime powers " << product.str() << " != " << order.str();
    }
}

TEST(gf2_field, equality_includes_primitive_element) {
    GF2Field d(8, GF2Poly::from_u64(0x11B));
    GF2Field e(8, GF2Poly::from_u64(0x11B), GF2Poly::from_u64(3));  // 3 is the default for 0x11B.
    GF2Field c(8, GF2Poly::from_u64(0x11B), GF2Poly::from_u64(5));
    GF2Field c2(8, GF2Poly::from_u64(0x11B), GF2Poly::from_u64(5));
    ASSERT_FALSE(e.has_custom_primitive_element());
    ASSERT_TRUE(c.has_custom_primitive_element());
    ASSERT_TRUE(d == e);
    ASSERT_TRUE(c == c2);
    ASSERT_TRUE(d != c);
    ASSERT_TRUE(d != GF2Field(8));
    ASSERT_TRUE(d != GF2Field(4));
}
