#include "gf2_field.h"

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
    // These must keep matching the values the original implementation hardcoded.
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
    ASSERT_EQ(gf2_default_irreducible_poly(12), GF2Poly::from_u64(4179));

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
