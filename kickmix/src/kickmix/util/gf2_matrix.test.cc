#include "gf2_matrix.h"

#include "gtest/gtest.h"

#include "test.test.h"

using namespace kickmix;

static GF2Matrix random_matrix(std::mt19937_64 &rng, size_t rows, size_t cols) {
    GF2Matrix result(rows, cols);
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            result.set(r, c, rng() & 1);
        }
    }
    return result;
}

static GF2Matrix random_invertible_matrix(std::mt19937_64 &rng, size_t n) {
    while (true) {
        GF2Matrix m = random_matrix(rng, n, n);
        if (m.rank() == n) {
            return m;
        }
    }
}

TEST(gf2_matrix, empty) {
    GF2Matrix m;
    ASSERT_EQ(m.rows, 0);
    ASSERT_EQ(m.cols, 0);
    ASSERT_EQ(m.rank(), 0);
}

TEST(gf2_matrix, get_set) {
    GF2Matrix m(3, 100);
    ASSERT_EQ(m.row_words, 2);
    ASSERT_FALSE(m.get(1, 70));
    m.set(1, 70, true);
    ASSERT_TRUE(m.get(1, 70));
    ASSERT_FALSE(m.get(0, 70));
    ASSERT_FALSE(m.get(1, 69));
    m.set(1, 70, false);
    ASSERT_FALSE(m.get(1, 70));
    m.xor_bit(2, 99);
    ASSERT_TRUE(m.get(2, 99));
    m.xor_bit(2, 99);
    ASSERT_FALSE(m.get(2, 99));
}

TEST(gf2_matrix, identity) {
    GF2Matrix m = GF2Matrix::identity(4);
    for (size_t r = 0; r < 4; r++) {
        for (size_t c = 0; c < 4; c++) {
            ASSERT_EQ(m.get(r, c), r == c);
        }
    }
    ASSERT_EQ(m.rank(), 4);
    ASSERT_EQ(m.inverse(), m);
    ASSERT_EQ(m.transposed(), m);
}

TEST(gf2_matrix, row_ops) {
    GF2Matrix m(2, 70);
    m.set(0, 5, true);
    m.set(0, 65, true);
    m.set(1, 5, true);
    ASSERT_FALSE(m.is_row_zero(0));
    m.xor_row_into(0, 1);
    ASSERT_FALSE(m.get(1, 5));
    ASSERT_TRUE(m.get(1, 65));
    m.swap_rows(0, 1);
    ASSERT_FALSE(m.get(0, 5));
    ASSERT_TRUE(m.get(1, 5));

    GF2Matrix z(1, 10);
    ASSERT_TRUE(z.is_row_zero(0));
}

TEST(gf2_matrix, multiply_identity) {
    auto rng = INDEPENDENT_TEST_RNG();
    GF2Matrix m = random_matrix(rng, 20, 20);
    GF2Matrix id = GF2Matrix::identity(20);
    ASSERT_EQ(m * id, m);
    ASSERT_EQ(id * m, m);
}

TEST(gf2_matrix, multiply_matches_naive) {
    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t trial = 0; trial < 30; trial++) {
        size_t a = 1 + rng() % 40;
        size_t b = 1 + rng() % 140;
        size_t c = 1 + rng() % 40;
        GF2Matrix lhs = random_matrix(rng, a, b);
        GF2Matrix rhs = random_matrix(rng, b, c);

        GF2Matrix expected(a, c);
        for (size_t i = 0; i < a; i++) {
            for (size_t j = 0; j < c; j++) {
                bool acc = false;
                for (size_t k = 0; k < b; k++) {
                    acc ^= lhs.get(i, k) && rhs.get(k, j);
                }
                expected.set(i, j, acc);
            }
        }
        ASSERT_EQ(lhs * rhs, expected) << a << " " << b << " " << c;
    }
}

TEST(gf2_matrix, multiply_dimension_mismatch) {
    GF2Matrix a(2, 3);
    GF2Matrix b(4, 5);
    ASSERT_THROW({ a *b; }, std::invalid_argument);
}

TEST(gf2_matrix, multiply_vector) {
    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t trial = 0; trial < 30; trial++) {
        size_t rows = 1 + rng() % 130;
        size_t cols = 1 + rng() % 130;
        GF2Matrix m = random_matrix(rng, rows, cols);
        GF2Poly v;
        for (size_t k = 0; k < cols; k++) {
            v.set_bit(k, rng() & 1);
        }

        GF2Poly expected;
        for (size_t r = 0; r < rows; r++) {
            bool acc = false;
            for (size_t c = 0; c < cols; c++) {
                acc ^= m.get(r, c) && v.bit(c);
            }
            expected.set_bit(r, acc);
        }
        ASSERT_EQ(m * v, expected) << rows << " " << cols;
    }
}

TEST(gf2_matrix, multiply_vector_is_linear) {
    auto rng = INDEPENDENT_TEST_RNG();
    GF2Matrix m = random_matrix(rng, 64, 64);
    GF2Poly a;
    GF2Poly b;
    for (size_t k = 0; k < 64; k++) {
        a.set_bit(k, rng() & 1);
        b.set_bit(k, rng() & 1);
    }
    ASSERT_EQ(m * (a ^ b), (m * a) ^ (m * b));
    ASSERT_TRUE((m * GF2Poly()).is_zero());
}

TEST(gf2_matrix, transposed) {
    auto rng = INDEPENDENT_TEST_RNG();
    GF2Matrix m = random_matrix(rng, 17, 100);
    GF2Matrix t = m.transposed();
    ASSERT_EQ(t.rows, 100);
    ASSERT_EQ(t.cols, 17);
    for (size_t r = 0; r < 17; r++) {
        for (size_t c = 0; c < 100; c++) {
            ASSERT_EQ(m.get(r, c), t.get(c, r));
        }
    }
    ASSERT_EQ(t.transposed(), m);
}

TEST(gf2_matrix, rank) {
    GF2Matrix m(3, 3);
    ASSERT_EQ(m.rank(), 0);
    ASSERT_EQ(GF2Matrix::identity(3).rank(), 3);

    // Two identical rows give rank 1.
    GF2Matrix d(2, 2);
    d.set(0, 0, true);
    d.set(1, 0, true);
    ASSERT_EQ(d.rank(), 1);
}

TEST(gf2_matrix, inverse) {
    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t n : std::vector<size_t>{1, 2, 3, 8, 33, 64, 65, 100}) {
        GF2Matrix m = random_invertible_matrix(rng, n);
        GF2Matrix inv = m.inverse();
        ASSERT_EQ(inv.rows, n);
        ASSERT_EQ(m * inv, GF2Matrix::identity(n)) << n;
        ASSERT_EQ(inv * m, GF2Matrix::identity(n)) << n;
    }
}

TEST(gf2_matrix, inverse_of_singular_is_empty) {
    GF2Matrix m(3, 3);
    ASSERT_EQ(m.inverse().rows, 0);
    GF2Matrix non_square(2, 3);
    ASSERT_EQ(non_square.inverse().rows, 0);
}

TEST(gf2_matrix, plu_decompose) {
    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t n : std::vector<size_t>{1, 2, 3, 8, 33, 64, 65, 130}) {
        GF2Matrix m = random_invertible_matrix(rng, n);
        GF2Matrix p;
        GF2Matrix l;
        GF2Matrix u;
        ASSERT_TRUE(m.plu_decompose(p, l, u)) << n;

        // The factors must have the promised shapes.
        for (size_t r = 0; r < n; r++) {
            ASSERT_TRUE(l.get(r, r)) << n;
            ASSERT_TRUE(u.get(r, r)) << n;
            for (size_t c = r + 1; c < n; c++) {
                ASSERT_FALSE(l.get(r, c)) << n;
                ASSERT_FALSE(u.get(c, r)) << n;
            }
        }
        // P must be a permutation matrix.
        for (size_t r = 0; r < n; r++) {
            size_t row_count = 0;
            size_t col_count = 0;
            for (size_t c = 0; c < n; c++) {
                row_count += p.get(r, c);
                col_count += p.get(c, r);
            }
            ASSERT_EQ(row_count, 1) << n;
            ASSERT_EQ(col_count, 1) << n;
        }
        // And the product must reconstruct the original.
        ASSERT_EQ(p * (l * u), m) << n;
    }
}

TEST(gf2_matrix, plu_decompose_rejects_bad_input) {
    GF2Matrix p;
    GF2Matrix l;
    GF2Matrix u;
    GF2Matrix singular(3, 3);
    ASSERT_FALSE(singular.plu_decompose(p, l, u));
    GF2Matrix non_square(2, 3);
    ASSERT_FALSE(non_square.plu_decompose(p, l, u));
}

TEST(gf2_matrix, str) {
    GF2Matrix m(2, 3);
    m.set(0, 0, true);
    m.set(1, 2, true);
    ASSERT_EQ(m.str(), "1..\n..1");
}
