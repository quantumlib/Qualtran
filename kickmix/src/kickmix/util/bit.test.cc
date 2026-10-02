#include "kickmix/util/bit.h"

#include "gtest/gtest.h"

#include "test.test.h"

using namespace kickmix;

TEST(bit, bit_length) {
    ASSERT_EQ(bit_length(0), 0);
    ASSERT_EQ(bit_length(1), 1);
    ASSERT_EQ(bit_length(2), 2);
    ASSERT_EQ(bit_length(3), 2);
    ASSERT_EQ(bit_length(4), 3);
    ASSERT_EQ(bit_length(5), 3);
    ASSERT_EQ(bit_length(6), 3);
    ASSERT_EQ(bit_length(7), 3);
    ASSERT_EQ(bit_length(8), 4);
    ASSERT_EQ(bit_length(9), 4);
    ASSERT_EQ(bit_length(10), 4);
    ASSERT_EQ(bit_length(11), 4);
    ASSERT_EQ(bit_length(12), 4);
    ASSERT_EQ(bit_length(13), 4);
    ASSERT_EQ(bit_length(14), 4);
    ASSERT_EQ(bit_length(15), 4);
    ASSERT_EQ(bit_length(16), 5);
    ASSERT_EQ(bit_length(17), 5);

    for (size_t k = 2; k < 64; k++) {
        ASSERT_EQ(bit_length((1ULL << k) + 1ULL), k + 1) << k;
        ASSERT_EQ(bit_length(1ULL << k), k + 1) << k;
        ASSERT_EQ(bit_length((1ULL << k) - 1ULL), k) << k;
    }

    ASSERT_EQ(bit_length(0b101ULL), 3);
    ASSERT_EQ(bit_length(0b1010101010101ULL), 13);
    ASSERT_EQ(bit_length(0b10101010101010101010101ULL), 23);
    ASSERT_EQ(bit_length(0b101010101010101010101010101010101ULL), 33);
    ASSERT_EQ(bit_length(0b1010101010101010101010101010101010101010101ULL), 43);
    ASSERT_EQ(bit_length(0b10101010101010101010101010101010101010101010101010101ULL), 53);
    ASSERT_EQ(bit_length(0b101010101010101010101010101010101010101010101010101010101010101ULL), 63);
}
