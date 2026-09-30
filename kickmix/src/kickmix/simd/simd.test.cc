#include "kickmix/simd/simd.test.h"

#include "gtest/gtest.h"

#include "test.test.h"

using namespace kickmix;

TEST_EACH_SIZE_W(val, non_zero, {
    W val{};
    for (size_t k = 0; k < val.BIT_SIZE; k++) {
        W val2 = val;
        ASSERT_FALSE(val2.bit(k));
        ASSERT_FALSE(val2.non_zero());

        val2.set_bit(k, true);
        ASSERT_TRUE(val2.bit(k));
        ASSERT_TRUE(val2.non_zero());

        val2.set_bit(k, false);
        ASSERT_FALSE(val2.bit(k));
        ASSERT_FALSE(val2.non_zero());
    }
})

TEST_EACH_SIZE_W(val, random, {
    W val{};
    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t k = 0; k < 100; k++) {
        val |= decltype(val)::random(rng);
    }
    val = ~val;
    ASSERT_FALSE(val.non_zero());
})

TEST_EACH_SIZE_W(val, randomize, {
    W val{};
    W val2{};
    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t k = 0; k < 100; k++) {
        val2.randomize(rng);
        val |= val2;
    }
    val = ~val;
    ASSERT_FALSE(val.non_zero());
})

TEST_EACH_SIZE_W(val, u32, {
    W val{};
    val = ~val;
    for (size_t k = 0; k < val.BIT_SIZE; k += 32) {
        ASSERT_EQ(val.u32(k / 32), UINT32_MAX);
    }
})

TEST_EACH_SIZE_W(val, clear_to_zero, {
    W val{};
    auto rng = INDEPENDENT_TEST_RNG();
    val.randomize(rng);
    val.clear_to_zero();
    for (size_t k = 0; k < val.BIT_SIZE; k += 32) {
        ASSERT_EQ(val.u32(k / 32), 0);
    }
    ASSERT_FALSE(val.non_zero());
})

TEST_EACH_SIZE_W(val, clear_to_max, {
    W val{};
    auto rng = INDEPENDENT_TEST_RNG();
    val.randomize(rng);
    val.clear_to_max();
    for (size_t k = 0; k < val.BIT_SIZE; k += 32) {
        ASSERT_EQ(val.u32(k / 32), UINT32_MAX);
    }
})

TEST_EACH_SIZE_W(val, ixor, {
    auto rng = INDEPENDENT_TEST_RNG();
    W val1 = W::random(rng);
    W val2 = W::random(rng);
    W val3 = val2;
    val3 ^= val1;
    for (size_t k = 0; k < val1.BIT_SIZE; k += 1) {
        ASSERT_EQ(val1.bit(k) ^ val2.bit(k), val3.bit(k));
    }
})

TEST_EACH_SIZE_W(val, iand, {
    auto rng = INDEPENDENT_TEST_RNG();
    W val1 = W::random(rng);
    W val2 = W::random(rng);
    W val3 = val2;
    val3 &= val1;
    for (size_t k = 0; k < val1.BIT_SIZE; k += 1) {
        ASSERT_EQ(val1.bit(k) & val2.bit(k), val3.bit(k));
    }
})

TEST_EACH_SIZE_W(val, ior, {
    auto rng = INDEPENDENT_TEST_RNG();
    W val1 = W::random(rng);
    W val2 = W::random(rng);
    W val3 = val2;
    val3 |= val1;
    for (size_t k = 0; k < val1.BIT_SIZE; k += 1) {
        ASSERT_EQ(val1.bit(k) | val2.bit(k), val3.bit(k));
    }
})

TEST_EACH_SIZE_W(val, oxor, {
    auto rng = INDEPENDENT_TEST_RNG();
    W val1 = W::random(rng);
    W val2 = W::random(rng);
    W val3 = val2 ^ val1;
    for (size_t k = 0; k < val1.BIT_SIZE; k += 1) {
        ASSERT_EQ(val1.bit(k) ^ val2.bit(k), val3.bit(k));
    }
})

TEST_EACH_SIZE_W(val, oand, {
    auto rng = INDEPENDENT_TEST_RNG();
    W val1 = W::random(rng);
    W val2 = W::random(rng);
    W val3 = val2 & val1;
    for (size_t k = 0; k < val1.BIT_SIZE; k += 1) {
        ASSERT_EQ(val1.bit(k) & val2.bit(k), val3.bit(k));
    }
})

TEST_EACH_SIZE_W(val, oor, {
    auto rng = INDEPENDENT_TEST_RNG();
    W val1 = W::random(rng);
    W val2 = W::random(rng);
    W val3 = val2 | val1;
    for (size_t k = 0; k < val1.BIT_SIZE; k += 1) {
        ASSERT_EQ(val1.bit(k) | val2.bit(k), val3.bit(k));
    }
})

TEST_EACH_SIZE_W(val, bitwise_negate, {
    auto rng = INDEPENDENT_TEST_RNG();
    W val1 = W::random(rng);
    W val3 = ~val1;
    for (size_t k = 0; k < val1.BIT_SIZE; k += 1) {
        ASSERT_EQ(val1.bit(k), !val3.bit(k));
    }
})

TEST_EACH_SIZE_W(val, from_u32_broadcast, {
    auto rng = INDEPENDENT_TEST_RNG();
    uint32_t u32 = static_cast<uint32_t>(rng());
    W val = W::from_u32_broadcast(u32);
    for (size_t k = 0; k < val.BIT_SIZE / 32; k += 1) {
        ASSERT_EQ(val.u32(k), u32);
    }
    for (size_t k = 0; k < val.BIT_SIZE / 64; k += 1) {
        ASSERT_EQ(val.u64(k), u32 | (static_cast<uint64_t>(u32) << 32));
    }
})

TEST_EACH_SIZE_W(val, u64_iadd, {
    auto rng = INDEPENDENT_TEST_RNG();
    auto val1 = W::random(rng);
    auto val2 = W::random(rng);
    auto val3 = val1;
    val3.u64_iadd(val2);
    for (size_t k = 0; k < val1.BIT_SIZE / 64; k += 1) {
        ASSERT_EQ(val3.u64(k), val1.u64(k) + val2.u64(k));
    }
})

TEST_EACH_SIZE_W(val, u64_right_shift, {
    auto rng = INDEPENDENT_TEST_RNG();
    auto val1 = W::random(rng);
    auto shift = rng() & 63;
    auto val3 = val1.u64_right_shift(shift);
    for (size_t k = 0; k < val1.BIT_SIZE / 64; k += 1) {
        ASSERT_EQ(val3.u64(k), val1.u64(k) >> shift);
    }
})

TEST_EACH_SIZE_W(val, u64_left_shift, {
    auto rng = INDEPENDENT_TEST_RNG();
    auto val1 = W::random(rng);
    auto shift = rng() & 63;
    auto val3 = val1.u64_left_shift(shift);
    for (size_t k = 0; k < val1.BIT_SIZE / 64; k += 1) {
        ASSERT_EQ(val3.u64(k), val1.u64(k) << shift);
    }
})

TEST_EACH_SIZE_W(val, u64_add, {
    auto rng = INDEPENDENT_TEST_RNG();
    auto val1 = W::random(rng);
    auto val2 = W::random(rng);
    auto val3 = val1.u64_add(val2);
    for (size_t k = 0; k < val1.BIT_SIZE / 64; k += 1) {
        ASSERT_EQ(val3.u64(k), val1.u64(k) + val2.u64(k));
    }
})

TEST_EACH_SIZE_W(val, equate, {
    auto rng = INDEPENDENT_TEST_RNG();
    auto val1 = W::random(rng);
    {
        auto val2 = val1;
        ASSERT_TRUE(val1 == val2);
        ASSERT_FALSE(val1 != val2);
    }
    for (size_t k = 0; k < val1.BIT_SIZE; k += 1) {
        auto val2 = val1;
        val2.set_bit(k, !val2.bit(k));
        ASSERT_FALSE(val1 == val2);
        ASSERT_TRUE(val1 != val2);
    }
})

TEST_EACH_SIZE_W(val, u32_eq, {
    std::array<uint32_t, W::BIT_SIZE / 32> v1;
    std::array<uint32_t, W::BIT_SIZE / 32> v2;
    std::array<uint32_t, W::BIT_SIZE / 32> v3;
    size_t n = v3.size();

    for (size_t k = 0; k < n; k++) {
        v1[k] = 0;
        v2[k] = 0;
        v3[k] = UINT32_MAX;
    }
    ASSERT_EQ((std::bit_cast<W>(v1).u32_eq(std::bit_cast<W>(v2))), (std::bit_cast<W>(v3)));

    for (size_t k = 0; k < n; k++) {
        v1[k] = 1;
        v2[k] = 0;
        v3[k] = 0;
    }
    ASSERT_EQ((std::bit_cast<W>(v1).u32_eq(std::bit_cast<W>(v2))), (std::bit_cast<W>(v3)));

    for (size_t k = 0; k < n; k++) {
        v1[k] = 0;
        v2[k] = 1;
        v3[k] = 0;
    }
    ASSERT_EQ((std::bit_cast<W>(v1).u32_eq(std::bit_cast<W>(v2))), (std::bit_cast<W>(v3)));

    for (size_t k = 0; k < n; k++) {
        v1[k] = 5;
        v2[k] = 5;
        v3[k] = UINT32_MAX;
    }
    ASSERT_EQ((std::bit_cast<W>(v1).u32_eq(std::bit_cast<W>(v2))), (std::bit_cast<W>(v3)));

    for (size_t k2 = 0; k2 <= n; k2++) {
        for (size_t k = 0; k < n; k++) {
            v1[k] = k;
            v2[k] = k2;
            v3[k] = k == k2 ? UINT32_MAX : 0;
        }
        ASSERT_EQ((std::bit_cast<W>(v1).u32_eq(std::bit_cast<W>(v2))), (std::bit_cast<W>(v3)));
    }

    for (size_t k = 0; k < n; k++) {
        v1[k] = k;
        v2[k] = 0;
        v3[k] = k == 0 ? UINT32_MAX : 0;
    }
    ASSERT_EQ((std::bit_cast<W>(v1).u32_eq(std::bit_cast<W>(v2))), (std::bit_cast<W>(v3)));

    for (size_t k = 0; k < n; k++) {
        v1[k] = UINT32_MAX;
        v2[k] = UINT32_MAX;
        v3[k] = UINT32_MAX;
    }
    ASSERT_EQ((std::bit_cast<W>(v1).u32_eq(std::bit_cast<W>(v2))), (std::bit_cast<W>(v3)));
});
