#include "kickmix/util/word_ops.h"

#include <array>
#include <bit>

#include "gtest/gtest.h"

#include "test.test.h"

using namespace kickmix;

TEST(word_ops, add_carry_u64) {
    std::array<uint64_t, 6> values{0, 1, 2, UINT64_MAX / 2, UINT64_MAX - 1, UINT64_MAX};
    for (uint64_t a : values) {
        for (uint64_t b : values) {
            for (bool carry : {false, true}) {
                unsigned __int128 expected = static_cast<unsigned __int128>(a) + b + carry;
                unsigned long long actual;
                bool carry_out = add_carry_u64(carry, a, b, &actual);
                ASSERT_EQ(actual, static_cast<uint64_t>(expected));
                ASSERT_EQ(carry_out, static_cast<bool>(expected >> 64));
            }
        }
    }
}

TEST(word_ops, sub_borrow_u64) {
    std::array<uint64_t, 6> values{0, 1, 2, UINT64_MAX / 2, UINT64_MAX - 1, UINT64_MAX};
    for (uint64_t a : values) {
        for (uint64_t b : values) {
            for (bool borrow : {false, true}) {
                unsigned __int128 subtrahend = static_cast<unsigned __int128>(b) + borrow;
                unsigned long long actual;
                bool borrow_out = sub_borrow_u64(borrow, a, b, &actual);
                ASSERT_EQ(actual, static_cast<uint64_t>(a - subtrahend));
                ASSERT_EQ(borrow_out, a < subtrahend);
            }
        }
    }
}

TEST(word_ops, extract_bits_u64) {
    ASSERT_EQ(extract_bits_u64(UINT64_MAX, 0), 0);
    ASSERT_EQ(extract_bits_u64(0, UINT64_MAX), 0);
    ASSERT_EQ(extract_bits_u64(UINT64_MAX, UINT64_MAX), UINT64_MAX);
    ASSERT_EQ(extract_bits_u64(0x9876543210ABCDEF, UINT64_MAX), 0x9876543210ABCDEF);
    ASSERT_EQ(extract_bits_u64(0x8000000000000000, 0x8000000000000001), 2);
    ASSERT_EQ(extract_bits_u64(1, 0x8000000000000001), 1);
    ASSERT_EQ(extract_bits_u64(0b110101, 0b101010), 0b100);

    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t k = 0; k < 1000; k++) {
        uint64_t value = rng();
        uint64_t value2 = rng();
        uint64_t mask = rng();
        uint64_t low_mask = mask & (UINT64_MAX >> (k % 64));
        size_t low_count = std::popcount(low_mask);
        uint64_t expected_low = low_count == 64 ? UINT64_MAX : (uint64_t{1} << low_count) - 1;

        uint64_t extracted = extract_bits_u64(value, mask);
        ASSERT_EQ(std::popcount(extracted), std::popcount(value & mask));
        ASSERT_EQ(extract_bits_u64(value & mask, mask), extracted);
        ASSERT_EQ(extract_bits_u64(value ^ value2, mask), extracted ^ extract_bits_u64(value2, mask));
        ASSERT_EQ(extract_bits_u64(low_mask, mask), expected_low);
    }
}
