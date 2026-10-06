#include "gf2_poly.h"

#include "gtest/gtest.h"

#include "test.test.h"

using namespace kickmix;

/// Naive reference carry-less multiply: shift and xor, one bit at a time.
static GF2Poly ref_mul(const GF2Poly &a, const GF2Poly &b) {
    GF2Poly result;
    size_t b_deg = b.degree();
    if (b_deg == SIZE_MAX || a.degree() == SIZE_MAX) {
        return result;
    }
    for (size_t k = 0; k <= b_deg; k++) {
        if (b.bit(k)) {
            result.ixor_shifted(a, k);
        }
    }
    return result;
}

static GF2Poly random_poly(std::mt19937_64 &rng, size_t num_bits) {
    GF2Poly result;
    for (size_t k = 0; k < num_bits; k++) {
        result.set_bit(k, rng() & 1);
    }
    return result;
}

TEST(gf2_poly, empty) {
    GF2Poly v;
    ASSERT_TRUE(v.is_zero());
    ASSERT_EQ(v.degree(), SIZE_MAX);
    ASSERT_EQ(v.num_bits_in_use(), 0);
    ASSERT_EQ(v.weight(), 0);
    ASSERT_FALSE(v.bit(0));
    ASSERT_FALSE(v.bit(GF2Poly::MAX_BITS * 4));
    ASSERT_EQ(v.str(), "0x0");
}

TEST(gf2_poly, from_u64) {
    GF2Poly v = GF2Poly::from_u64(0x11B);
    ASSERT_FALSE(v.is_zero());
    ASSERT_EQ(v.degree(), 8);
    ASSERT_EQ(v.num_bits_in_use(), 9);
    ASSERT_EQ(v.weight(), 5);
    ASSERT_TRUE(v.bit(0));
    ASSERT_TRUE(v.bit(1));
    ASSERT_FALSE(v.bit(2));
    ASSERT_TRUE(v.bit(3));
    ASSERT_TRUE(v.bit(4));
    ASSERT_TRUE(v.bit(8));
    ASSERT_EQ(v.str(), "0x11B");
}

TEST(gf2_poly, monomial) {
    for (size_t k : std::vector<size_t>{0, 1, 63, 64, 65, 511, 512, GF2Poly::MAX_BITS - 1}) {
        GF2Poly v = GF2Poly::monomial(k);
        ASSERT_EQ(v.degree(), k) << k;
        ASSERT_EQ(v.weight(), 1) << k;
        ASSERT_TRUE(v.bit(k)) << k;
    }
    ASSERT_THROW({ GF2Poly::monomial(GF2Poly::MAX_BITS); }, std::invalid_argument);
}

TEST(gf2_poly, set_and_xor_bit) {
    GF2Poly v;
    v.set_bit(100, true);
    ASSERT_TRUE(v.bit(100));
    v.set_bit(100, false);
    ASSERT_FALSE(v.bit(100));
    v.xor_bit(200);
    ASSERT_TRUE(v.bit(200));
    v.xor_bit(200);
    ASSERT_FALSE(v.bit(200));
    ASSERT_TRUE(v.is_zero());
    ASSERT_THROW({ v.set_bit(GF2Poly::MAX_BITS, true); }, std::invalid_argument);
    ASSERT_THROW({ v.xor_bit(GF2Poly::MAX_BITS); }, std::invalid_argument);
}

TEST(gf2_poly, truncate) {
    GF2Poly v = GF2Poly::from_u64(0xFF);
    v.truncate(4);
    ASSERT_EQ(v, GF2Poly::from_u64(0xF));

    GF2Poly w = GF2Poly::monomial(200);
    w ^= GF2Poly::monomial(100);
    w.truncate(150);
    ASSERT_EQ(w, GF2Poly::monomial(100));

    GF2Poly u = GF2Poly::monomial(64);
    u.truncate(64);
    ASSERT_TRUE(u.is_zero());
}

TEST(gf2_poly, xor) {
    GF2Poly a = GF2Poly::from_u64(0b1100);
    GF2Poly b = GF2Poly::from_u64(0b1010);
    ASSERT_EQ(a ^ b, GF2Poly::from_u64(0b0110));
    ASSERT_EQ(a ^ a, GF2Poly());
    a ^= b;
    ASSERT_EQ(a, GF2Poly::from_u64(0b0110));
}

TEST(gf2_poly, shifts) {
    GF2Poly a = GF2Poly::from_u64(0b1011);
    ASSERT_EQ(a << 1, GF2Poly::from_u64(0b10110));
    ASSERT_EQ(a << 0, a);
    ASSERT_EQ((a << 64) >> 64, a);
    ASSERT_EQ((a << 3) >> 3, a);
    ASSERT_EQ((a << 65) >> 65, a);
    ASSERT_EQ((a << 100) >> 100, a);
    ASSERT_TRUE((a >> 4).is_zero());
    ASSERT_TRUE((a << GF2Poly::MAX_BITS).is_zero());
    ASSERT_TRUE((a >> GF2Poly::MAX_BITS).is_zero());

    // Overflowing coefficients are discarded.
    GF2Poly b = GF2Poly::from_u64(0b1011);
    b <<= GF2Poly::MAX_BITS - 3;
    ASSERT_EQ(b.degree(), GF2Poly::MAX_BITS - 2);
    ASSERT_EQ(b.weight(), 2);
}

TEST(gf2_poly, shifts_fuzz) {
    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t trial = 0; trial < 200; trial++) {
        size_t n = 1 + rng() % 400;
        size_t shift = rng() % 400;
        GF2Poly a = random_poly(rng, n);
        ASSERT_EQ((a << shift) >> shift, a) << n << " " << shift;

        GF2Poly expected;
        for (size_t k = 0; k < n; k++) {
            if (a.bit(k)) {
                expected.set_bit(k + shift, true);
            }
        }
        ASSERT_EQ(a << shift, expected) << n << " " << shift;
    }
}

TEST(gf2_poly, ixor_shifted) {
    GF2Poly a = GF2Poly::from_u64(0b1);
    a.ixor_shifted(GF2Poly::from_u64(0b11), 4);
    ASSERT_EQ(a, GF2Poly::from_u64(0b110001));

    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t trial = 0; trial < 200; trial++) {
        GF2Poly base = random_poly(rng, 1 + rng() % 500);
        GF2Poly add = random_poly(rng, 1 + rng() % 500);
        size_t shift = rng() % 500;
        GF2Poly got = base;
        got.ixor_shifted(add, shift);
        ASSERT_EQ(got, base ^ (add << shift)) << shift;
    }
}

TEST(gf2_poly, mul_small) {
    ASSERT_TRUE(GF2Poly::mul(GF2Poly(), GF2Poly::from_u64(5)).is_zero());
    ASSERT_TRUE(GF2Poly::mul(GF2Poly::from_u64(5), GF2Poly()).is_zero());
    ASSERT_EQ(GF2Poly::mul(GF2Poly::from_u64(5), GF2Poly::from_u64(1)), GF2Poly::from_u64(5));
    // (x + 1) * (x + 1) = x^2 + 1 over GF(2).
    ASSERT_EQ(GF2Poly::mul(GF2Poly::from_u64(0b11), GF2Poly::from_u64(0b11)), GF2Poly::from_u64(0b101));
    // (x^2 + x + 1) * (x + 1) = x^3 + 1 over GF(2).
    ASSERT_EQ(GF2Poly::mul(GF2Poly::from_u64(0b111), GF2Poly::from_u64(0b11)), GF2Poly::from_u64(0b1001));
    // Multiplying by a monomial is a shift.
    GF2Poly a = GF2Poly::from_u64(0xDEADBEEF);
    ASSERT_EQ(GF2Poly::mul(a, GF2Poly::monomial(37)), a << 37);
}

TEST(gf2_poly, mul_matches_reference_fuzz) {
    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t trial = 0; trial < 500; trial++) {
        size_t na = 1 + rng() % GF2_MAX_DEGREE;
        size_t nb = 1 + rng() % GF2_MAX_DEGREE;
        GF2Poly a = random_poly(rng, na);
        GF2Poly b = random_poly(rng, nb);
        GF2Poly expected = ref_mul(a, b);
        ASSERT_EQ(GF2Poly::mul(a, b), expected) << na << " " << nb;
        // Multiplication is commutative.
        ASSERT_EQ(GF2Poly::mul(b, a), expected) << na << " " << nb;
    }
}

TEST(gf2_poly, mul_full_width) {
    // Two degree 511 polynomials produce a degree 1022 product, which must fit.
    GF2Poly a = GF2Poly::monomial(511);
    a ^= GF2Poly::from_u64(1);
    GF2Poly b = GF2Poly::monomial(511);
    b ^= GF2Poly::from_u64(1);
    GF2Poly p = GF2Poly::mul(a, b);
    ASSERT_EQ(p.degree(), 1022);
    ASSERT_EQ(p.weight(), 2);
    ASSERT_TRUE(p.bit(1022));
    ASSERT_TRUE(p.bit(0));
}

TEST(gf2_poly, mul_overflow_throws) {
    ASSERT_THROW(
        { GF2Poly::mul(GF2Poly::monomial(GF2Poly::MAX_BITS - 1), GF2Poly::monomial(1)); }, std::invalid_argument);
}

TEST(gf2_poly, imod) {
    GF2Poly f = GF2Poly::from_u64(0x11B);

    GF2Poly v = GF2Poly::from_u64(0x1FF);
    v.imod(f);
    ASSERT_EQ(v, GF2Poly::from_u64(0xE4));

    GF2Poly w = GF2Poly::from_u64(0xFF);
    w.imod(f);
    ASSERT_EQ(w, GF2Poly::from_u64(0xFF));

    GF2Poly z = f;
    z.imod(f);
    ASSERT_TRUE(z.is_zero());

    GF2Poly big = GF2Poly::monomial(1000);
    big.imod(f);
    ASSERT_LT(big.degree(), 8u);

    ASSERT_THROW({ v.imod(GF2Poly()); }, std::invalid_argument);
}

TEST(gf2_poly, imod_fuzz) {
    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t trial = 0; trial < 200; trial++) {
        size_t nf = 2 + rng() % 200;
        GF2Poly f = random_poly(rng, nf);
        f.set_bit(nf - 1, true);
        GF2Poly v = random_poly(rng, 1 + rng() % 600);

        GF2Poly r = v;
        r.imod(f);
        ASSERT_TRUE(r.is_zero() || r.degree() < f.degree());

        // v - r must be divisible by f, i.e. reducing it again gives zero.
        GF2Poly diff = v ^ r;
        diff.imod(f);
        ASSERT_TRUE(diff.is_zero());
    }
}

TEST(gf2_poly, gcd) {
    GF2Poly a = GF2Poly::from_u64(0b1011);
    ASSERT_EQ(GF2Poly::gcd(a, a), a);
    ASSERT_EQ(GF2Poly::gcd(a, GF2Poly()), a);
    ASSERT_EQ(GF2Poly::gcd(GF2Poly(), a), a);
    // gcd(x^3 + 1, x^2 + 1) = x + 1, since both are divisible by (x + 1).
    ASSERT_EQ(GF2Poly::gcd(GF2Poly::from_u64(0b1001), GF2Poly::from_u64(0b101)), GF2Poly::from_u64(0b11));
    // Coprime: gcd(x, x + 1) = 1.
    ASSERT_EQ(GF2Poly::gcd(GF2Poly::from_u64(0b10), GF2Poly::from_u64(0b11)), GF2Poly::from_u64(1));
}

TEST(gf2_poly, gcd_fuzz) {
    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t trial = 0; trial < 100; trial++) {
        GF2Poly a = random_poly(rng, 1 + rng() % 100);
        GF2Poly b = random_poly(rng, 1 + rng() % 100);
        if (a.is_zero() || b.is_zero()) {
            continue;
        }
        GF2Poly common = random_poly(rng, 1 + rng() % 20);
        if (common.is_zero()) {
            continue;
        }
        GF2Poly g = GF2Poly::gcd(GF2Poly::mul(a, common), GF2Poly::mul(b, common));
        // The gcd must be divisible by the planted common factor.
        GF2Poly check = g;
        check.imod(common);
        ASSERT_TRUE(check.is_zero());
    }
}

TEST(gf2_poly, invert_mod_gf256_exhaustive) {
    // The AES field polynomial x^8 + x^4 + x^3 + x + 1.
    GF2Poly f = GF2Poly::from_u64(0x11B);
    GF2Poly one = GF2Poly::from_u64(1);
    for (uint64_t v = 1; v < 256; v++) {
        GF2Poly a = GF2Poly::from_u64(v);
        GF2Poly inv = GF2Poly::invert_mod(a, f);
        ASSERT_FALSE(inv.is_zero()) << v;
        GF2Poly product = GF2Poly::mul(a, inv);
        product.imod(f);
        ASSERT_EQ(product, one) << v;
    }
    ASSERT_TRUE(GF2Poly::invert_mod(GF2Poly(), f).is_zero());
    ASSERT_THROW({ GF2Poly::invert_mod(GF2Poly::from_u64(1), GF2Poly()); }, std::invalid_argument);
}

TEST(gf2_poly, invert_mod_reducible_modulus) {
    // Modulo x^2, the element x is a zero divisor and has no inverse.
    ASSERT_TRUE(GF2Poly::invert_mod(GF2Poly::from_u64(0b10), GF2Poly::from_u64(0b100)).is_zero());
    // But x + 1 is invertible modulo x^2: (x + 1)^2 = x^2 + 1 == 1.
    GF2Poly inv = GF2Poly::invert_mod(GF2Poly::from_u64(0b11), GF2Poly::from_u64(0b100));
    ASSERT_EQ(inv, GF2Poly::from_u64(0b11));
}

TEST(gf2_poly, fixed_width_int_round_trip) {
    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t trial = 0; trial < 100; trial++) {
        size_t n = 1 + rng() % GF2Poly::MAX_BITS;
        GF2Poly a = random_poly(rng, n);
        FixedWidthInt f = a.to_fixed_width_int(n);
        ASSERT_EQ(f.num_bits, n);
        ASSERT_EQ(GF2Poly::from_fixed_width_int(f), a) << n;
    }
    ASSERT_THROW({ GF2Poly::from_u64(1).to_fixed_width_int(GF2Poly::MAX_BITS + 1); }, std::invalid_argument);
}

TEST(gf2_poly, str) {
    ASSERT_EQ(GF2Poly().str(), "0x0");
    ASSERT_EQ(GF2Poly::from_u64(1).str(), "0x1");
    ASSERT_EQ(GF2Poly::from_u64(0xDEADBEEF).str(), "0xDEADBEEF");
    ASSERT_EQ(GF2Poly::monomial(64).str(), "0x10000000000000000");
}

TEST(gf2_poly, ixor_shifted_self_aliasing) {
    for (size_t shift : std::vector<size_t>{0, 1, 13, 63, 64, 65, 128, 192}) {
        GF2Poly a;
        a.words[0] = 0xDEADBEEFCAFEBABEull;
        a.words[1] = 0x0123456789ABCDEFull;
        GF2Poly expected = a ^ (a << shift);
        a.ixor_shifted(a, shift);
        ASSERT_EQ(a, expected) << "shift=" << shift;
    }
}
