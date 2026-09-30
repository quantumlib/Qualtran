#include "kickmix/util/fixed_width_int.h"

#include "gtest/gtest.h"

#include "test.test.h"

using namespace kickmix;

TEST(FixedWidthInt, str) {
    ASSERT_EQ(FixedWidthInt(5, "0b0").str(), "0x00 [num_bits=5] (decimal=0)");
    ASSERT_EQ(FixedWidthInt(5, "0b00000").str(), "0x00 [num_bits=5] (decimal=0)");
    ASSERT_EQ(FixedWidthInt(5, "0b00001").str(), "0x01 [num_bits=5] (decimal=1)");
    ASSERT_EQ(FixedWidthInt(5, "0b00010").str(), "0x02 [num_bits=5] (decimal=2)");
    ASSERT_EQ(FixedWidthInt(5, "0b00100").str(), "0x04 [num_bits=5] (decimal=4)");
    ASSERT_EQ(FixedWidthInt(5, "0b01000").str(), "0x08 [num_bits=5] (decimal=8)");
    ASSERT_EQ(FixedWidthInt(5, "0b01_000").str(), "0x08 [num_bits=5] (decimal=8)");
    ASSERT_EQ(FixedWidthInt(5, "0b0_1000").str(), "0x08 [num_bits=5] (decimal=8)");
    ASSERT_EQ(FixedWidthInt(5, "0b10000").str(), "0x10 [num_bits=5] (decimal=16)");
    ASSERT_EQ(FixedWidthInt(5, "0b11111").str(), "0x1F [num_bits=5] (decimal=31)");
    ASSERT_THROW({ FixedWidthInt(5, "0b111111").str(); }, std::invalid_argument);
    ASSERT_THROW({ FixedWidthInt(5, "0xFFFFFFF").str(); }, std::invalid_argument);
    ASSERT_THROW({ FixedWidthInt(5, "0xFFFFFF2").str(); }, std::invalid_argument);
    ASSERT_THROW({ FixedWidthInt(8, "0xFFFFFF2").str(); }, std::invalid_argument);

    ASSERT_EQ(
        FixedWidthInt(
            100,
            "0b0_10000000_01000000_00100000_00001000_00000100_00000010_00000001_11000000_01100000_00110000_00011000_"
            "00001100")
            .str(),
        "0x080402008_040201C06030180C [num_bits=100] (decimal=39691603773177704734140667916)");
    ASSERT_EQ(
        FixedWidthInt(100, "0x080402008_040201C06030180C").str(),
        "0x080402008_040201C06030180C [num_bits=100] (decimal=39691603773177704734140667916)");
}

TEST(FixedWidthInt, hex_bin) {
    ASSERT_EQ(FixedWidthInt(5, 0b00000).bin(), "00000");
    ASSERT_EQ(FixedWidthInt(5, 0b11111).bin(), "11111");
    ASSERT_EQ(FixedWidthInt(15, 0b101010101010111).bin(), "1010101_01010111");
    ASSERT_EQ(FixedWidthInt(16, 0b101010101010101).bin(), "01010101_01010101");
    ASSERT_EQ(FixedWidthInt(17, 0b10101010101010101).bin(), "1_01010101_01010101");

    ASSERT_EQ(FixedWidthInt(5, 0x00).hex(), "00");
    ASSERT_EQ(FixedWidthInt(5, 0x1F).hex(), "1F");
    ASSERT_EQ(FixedWidthInt(60, uint64_t{0x123456789ABCDEF}).hex(), "1234567_89ABCDEF");
    ASSERT_EQ(FixedWidthInt(64, uint64_t{0xA123456789ABCDEF}).hex(), "A1234567_89ABCDEF");
    ASSERT_EQ(FixedWidthInt(33, uint64_t{0x189ABCDEF}).hex(), "1_89ABCDEF");
    ASSERT_EQ(FixedWidthInt(32, 0x89ABCDEF).hex(), "89ABCDEF");
}

TEST(FixedWidthInt, conv) {
    ASSERT_EQ(FixedWidthInt(0, static_cast<int64_t>(0)).str(), "0x [num_bits=0] (decimal=0)");
    ASSERT_EQ(FixedWidthInt(1, static_cast<int64_t>(1)).str(), "0x1 [num_bits=1] (decimal=1)");

    ASSERT_EQ(FixedWidthInt(71, static_cast<int64_t>(0)).str(), "0x00_0000000000000000 [num_bits=71] (decimal=0)");
    ASSERT_EQ(FixedWidthInt(71, static_cast<int64_t>(1)).str(), "0x00_0000000000000001 [num_bits=71] (decimal=1)");
    ASSERT_EQ(
        FixedWidthInt(71, static_cast<int64_t>(-1)).str(),
        "0x7F_FFFFFFFFFFFFFFFF [num_bits=71] (decimal=2361183241434822606847)");

    ASSERT_EQ(FixedWidthInt(71, static_cast<int32_t>(0)).str(), "0x00_0000000000000000 [num_bits=71] (decimal=0)");
    ASSERT_EQ(FixedWidthInt(71, static_cast<int32_t>(1)).str(), "0x00_0000000000000001 [num_bits=71] (decimal=1)");
    ASSERT_EQ(
        FixedWidthInt(71, static_cast<int32_t>(-1)).str(),
        "0x7F_FFFFFFFFFFFFFFFF [num_bits=71] (decimal=2361183241434822606847)");

    ASSERT_EQ(FixedWidthInt(71, static_cast<int16_t>(0)).str(), "0x00_0000000000000000 [num_bits=71] (decimal=0)");
    ASSERT_EQ(FixedWidthInt(71, static_cast<int16_t>(1)).str(), "0x00_0000000000000001 [num_bits=71] (decimal=1)");
    ASSERT_EQ(
        FixedWidthInt(71, static_cast<int16_t>(-1)).str(),
        "0x7F_FFFFFFFFFFFFFFFF [num_bits=71] (decimal=2361183241434822606847)");

    ASSERT_EQ(FixedWidthInt(71, static_cast<int8_t>(0)).str(), "0x00_0000000000000000 [num_bits=71] (decimal=0)");
    ASSERT_EQ(FixedWidthInt(71, static_cast<int8_t>(1)).str(), "0x00_0000000000000001 [num_bits=71] (decimal=1)");
    ASSERT_EQ(
        FixedWidthInt(71, static_cast<int8_t>(-1)).str(),
        "0x7F_FFFFFFFFFFFFFFFF [num_bits=71] (decimal=2361183241434822606847)");

    ASSERT_EQ(FixedWidthInt(71, static_cast<uint64_t>(0)).str(), "0x00_0000000000000000 [num_bits=71] (decimal=0)");
    ASSERT_EQ(FixedWidthInt(71, static_cast<uint64_t>(1)).str(), "0x00_0000000000000001 [num_bits=71] (decimal=1)");
    ASSERT_EQ(
        FixedWidthInt(71, static_cast<uint64_t>(-1)).str(),
        "0x00_FFFFFFFFFFFFFFFF [num_bits=71] (decimal=18446744073709551615)");

    ASSERT_EQ(FixedWidthInt(71, static_cast<uint32_t>(0)).str(), "0x00_0000000000000000 [num_bits=71] (decimal=0)");
    ASSERT_EQ(FixedWidthInt(71, static_cast<uint32_t>(1)).str(), "0x00_0000000000000001 [num_bits=71] (decimal=1)");
    ASSERT_EQ(
        FixedWidthInt(71, static_cast<uint32_t>(-1)).str(), "0x00_00000000FFFFFFFF [num_bits=71] (decimal=4294967295)");

    ASSERT_EQ(FixedWidthInt(71, static_cast<uint16_t>(0)).str(), "0x00_0000000000000000 [num_bits=71] (decimal=0)");
    ASSERT_EQ(FixedWidthInt(71, static_cast<uint16_t>(1)).str(), "0x00_0000000000000001 [num_bits=71] (decimal=1)");
    ASSERT_EQ(
        FixedWidthInt(71, static_cast<uint16_t>(-1)).str(), "0x00_000000000000FFFF [num_bits=71] (decimal=65535)");

    ASSERT_EQ(FixedWidthInt(71, static_cast<uint8_t>(0)).str(), "0x00_0000000000000000 [num_bits=71] (decimal=0)");
    ASSERT_EQ(FixedWidthInt(71, static_cast<uint8_t>(1)).str(), "0x00_0000000000000001 [num_bits=71] (decimal=1)");
    ASSERT_EQ(FixedWidthInt(71, static_cast<uint8_t>(-1)).str(), "0x00_00000000000000FF [num_bits=71] (decimal=255)");
}

TEST(FixedWidthInt, increment) {
    FixedWidthInt b(0, "0x");
    ASSERT_EQ(b.str(), "0x [num_bits=0] (decimal=0)");
    b.increment();
    ASSERT_EQ(b.str(), "0x [num_bits=0] (decimal=0)");

    b = FixedWidthInt(3, "0x6");
    ASSERT_EQ(b.str(), "0x6 [num_bits=3] (decimal=6)");
    b.increment();
    ASSERT_EQ(b.str(), "0x7 [num_bits=3] (decimal=7)");
    b.increment();
    ASSERT_EQ(b.str(), "0x0 [num_bits=3] (decimal=0)");
    b.increment();
    ASSERT_EQ(b.str(), "0x1 [num_bits=3] (decimal=1)");

    b = FixedWidthInt(71, "0x7F_FFFFFFFFFFFFFFFE");
    ASSERT_EQ(b.str(), "0x7F_FFFFFFFFFFFFFFFE [num_bits=71] (decimal=2361183241434822606846)");
    b.increment();
    ASSERT_EQ(b.str(), "0x7F_FFFFFFFFFFFFFFFF [num_bits=71] (decimal=2361183241434822606847)");
    b.increment();
    ASSERT_EQ(b.str(), "0x00_0000000000000000 [num_bits=71] (decimal=0)");
    b.increment();
    ASSERT_EQ(b.str(), "0x00_0000000000000001 [num_bits=71] (decimal=1)");
}

TEST(FixedWidthInt, decrement) {
    FixedWidthInt b(0, "0x");
    ASSERT_EQ(b.str(), "0x [num_bits=0] (decimal=0)");
    b.decrement();
    ASSERT_EQ(b.str(), "0x [num_bits=0] (decimal=0)");

    b = FixedWidthInt(3, "0x1");
    ASSERT_EQ(b.str(), "0x1 [num_bits=3] (decimal=1)");
    b.decrement();
    ASSERT_EQ(b.str(), "0x0 [num_bits=3] (decimal=0)");
    b.decrement();
    ASSERT_EQ(b.str(), "0x7 [num_bits=3] (decimal=7)");
    b.decrement();
    ASSERT_EQ(b.str(), "0x6 [num_bits=3] (decimal=6)");

    b = FixedWidthInt(71, "0x00_0000000000000001");
    ASSERT_EQ(b.str(), "0x00_0000000000000001 [num_bits=71] (decimal=1)");
    b.decrement();
    ASSERT_EQ(b.str(), "0x00_0000000000000000 [num_bits=71] (decimal=0)");
    b.decrement();
    ASSERT_EQ(b.str(), "0x7F_FFFFFFFFFFFFFFFF [num_bits=71] (decimal=2361183241434822606847)");
    b.decrement();
    ASSERT_EQ(b.str(), "0x7F_FFFFFFFFFFFFFFFE [num_bits=71] (decimal=2361183241434822606846)");
}

TEST(FixedWidthInt, eq_big_int) {
    ASSERT_EQ(FixedWidthInt(0, "0x"), FixedWidthInt(0, "0x"));
    ASSERT_EQ(FixedWidthInt(10, "0x"), FixedWidthInt(0, "0x"));
    ASSERT_EQ(FixedWidthInt(100, "0x"), FixedWidthInt(0, "0x"));
    ASSERT_EQ(FixedWidthInt(100, "0x"), FixedWidthInt(10, "0x"));

    ASSERT_NE(FixedWidthInt(100, "0x1"), FixedWidthInt(0, "0x"));
    ASSERT_NE(FixedWidthInt(100, "0x1"), FixedWidthInt(10, "0x"));
    ASSERT_NE(FixedWidthInt(10, "0x1"), FixedWidthInt(10, "0x"));

    ASSERT_EQ(FixedWidthInt(100, "0xFFFF"), FixedWidthInt(40, "0xFFFF"));
    ASSERT_NE(FixedWidthInt(100, "0xFFFF"), FixedWidthInt(40, "0xFFFE"));
    ASSERT_EQ(FixedWidthInt(100, "0x0_000000000000FFFF"), FixedWidthInt(40, "0xFFFF"));
    ASSERT_NE(FixedWidthInt(100, "0x1_000000000000FFFF"), FixedWidthInt(40, "0xFFFF"));
    ASSERT_EQ(FixedWidthInt(40, "0xFFFF"), FixedWidthInt(100, "0x0_000000000000FFFF"));
    ASSERT_NE(FixedWidthInt(40, "0xFFFF"), FixedWidthInt(100, "0x1_000000000000FFFF"));
}

TEST(FixedWidthInt, eq_int) {
    ASSERT_EQ(FixedWidthInt(0, "0x"), static_cast<uint64_t>(0));
    ASSERT_EQ(FixedWidthInt(0, "0x"), static_cast<uint32_t>(0));
    ASSERT_EQ(FixedWidthInt(0, "0x"), static_cast<uint16_t>(0));
    ASSERT_EQ(FixedWidthInt(0, "0x"), static_cast<uint8_t>(0));
    ASSERT_EQ(FixedWidthInt(0, "0x"), static_cast<int64_t>(0));
    ASSERT_EQ(FixedWidthInt(0, "0x"), static_cast<int32_t>(0));
    ASSERT_EQ(FixedWidthInt(0, "0x"), static_cast<int16_t>(0));
    ASSERT_EQ(FixedWidthInt(0, "0x"), static_cast<int8_t>(0));

    ASSERT_NE(FixedWidthInt(0, "0x"), static_cast<uint64_t>(1));
    ASSERT_NE(FixedWidthInt(0, "0x"), static_cast<uint32_t>(1));
    ASSERT_NE(FixedWidthInt(0, "0x"), static_cast<uint16_t>(1));
    ASSERT_NE(FixedWidthInt(0, "0x"), static_cast<uint8_t>(1));
    ASSERT_NE(FixedWidthInt(0, "0x"), static_cast<int64_t>(1));
    ASSERT_NE(FixedWidthInt(0, "0x"), static_cast<int32_t>(1));
    ASSERT_NE(FixedWidthInt(0, "0x"), static_cast<int16_t>(1));
    ASSERT_NE(FixedWidthInt(0, "0x"), static_cast<int8_t>(1));

    ASSERT_EQ(FixedWidthInt(64, "0xFFFFFFFFFFFFFFFF"), static_cast<uint64_t>(-1));
    ASSERT_NE(FixedWidthInt(64, "0xFFFFFFFFFFFFFFFF"), static_cast<uint32_t>(-1));
    ASSERT_NE(FixedWidthInt(64, "0xFFFFFFFFFFFFFFFF"), static_cast<uint16_t>(-1));
    ASSERT_NE(FixedWidthInt(64, "0xFFFFFFFFFFFFFFFF"), static_cast<uint8_t>(-1));
    ASSERT_NE(FixedWidthInt(64, "0xFFFFFFFFFFFFFFFF"), static_cast<int64_t>(-1));
    ASSERT_NE(FixedWidthInt(64, "0xFFFFFFFFFFFFFFFF"), static_cast<int32_t>(-1));
    ASSERT_NE(FixedWidthInt(64, "0xFFFFFFFFFFFFFFFF"), static_cast<int16_t>(-1));
    ASSERT_NE(FixedWidthInt(64, "0xFFFFFFFFFFFFFFFF"), static_cast<int8_t>(-1));

    ASSERT_EQ(FixedWidthInt(10, "0x0F"), static_cast<uint64_t>(0xF));
    ASSERT_EQ(FixedWidthInt(10, "0x0F"), static_cast<uint32_t>(0xF));
    ASSERT_EQ(FixedWidthInt(10, "0x0F"), static_cast<uint16_t>(0xF));
    ASSERT_EQ(FixedWidthInt(10, "0x0F"), static_cast<uint8_t>(0xF));
    ASSERT_EQ(FixedWidthInt(10, "0x0F"), static_cast<int64_t>(0xF));
    ASSERT_EQ(FixedWidthInt(10, "0x0F"), static_cast<int32_t>(0xF));
    ASSERT_EQ(FixedWidthInt(10, "0x0F"), static_cast<int16_t>(0xF));
    ASSERT_EQ(FixedWidthInt(10, "0x0F"), static_cast<int8_t>(0xF));

    ASSERT_NE(FixedWidthInt(10, "0x0F"), static_cast<uint64_t>(1));
    ASSERT_NE(FixedWidthInt(10, "0x0F"), static_cast<uint32_t>(1));
    ASSERT_NE(FixedWidthInt(10, "0x0F"), static_cast<uint16_t>(1));
    ASSERT_NE(FixedWidthInt(10, "0x0F"), static_cast<uint8_t>(1));
    ASSERT_NE(FixedWidthInt(10, "0x0F"), static_cast<int64_t>(1));
    ASSERT_NE(FixedWidthInt(10, "0x0F"), static_cast<int32_t>(1));
    ASSERT_NE(FixedWidthInt(10, "0x0F"), static_cast<int16_t>(1));
    ASSERT_NE(FixedWidthInt(10, "0x0F"), static_cast<int8_t>(1));

    ASSERT_NE(FixedWidthInt(100, "0x1_0000000000000002"), static_cast<uint64_t>(2));
    ASSERT_NE(FixedWidthInt(100, "0x1_0000000000000002"), static_cast<uint32_t>(2));
    ASSERT_NE(FixedWidthInt(100, "0x1_0000000000000002"), static_cast<uint16_t>(2));
    ASSERT_NE(FixedWidthInt(100, "0x1_0000000000000002"), static_cast<uint8_t>(2));
    ASSERT_NE(FixedWidthInt(100, "0x1_0000000000000002"), static_cast<int64_t>(2));
    ASSERT_NE(FixedWidthInt(100, "0x1_0000000000000002"), static_cast<int32_t>(2));
    ASSERT_NE(FixedWidthInt(100, "0x1_0000000000000002"), static_cast<int16_t>(2));
    ASSERT_NE(FixedWidthInt(100, "0x1_0000000000000002"), static_cast<int8_t>(2));

    ASSERT_EQ(FixedWidthInt(100, "0x0_0000000000000002"), static_cast<uint64_t>(2));
    ASSERT_EQ(FixedWidthInt(100, "0x0_0000000000000002"), static_cast<uint32_t>(2));
    ASSERT_EQ(FixedWidthInt(100, "0x0_0000000000000002"), static_cast<uint16_t>(2));
    ASSERT_EQ(FixedWidthInt(100, "0x0_0000000000000002"), static_cast<uint8_t>(2));
    ASSERT_EQ(FixedWidthInt(100, "0x0_0000000000000002"), static_cast<int64_t>(2));
    ASSERT_EQ(FixedWidthInt(100, "0x0_0000000000000002"), static_cast<int32_t>(2));
    ASSERT_EQ(FixedWidthInt(100, "0x0_0000000000000002"), static_cast<int16_t>(2));
    ASSERT_EQ(FixedWidthInt(100, "0x0_0000000000000002"), static_cast<int8_t>(2));

    ASSERT_EQ(FixedWidthInt(10, "0x2"), static_cast<uint64_t>(2));
    ASSERT_EQ(FixedWidthInt(10, "0x2"), static_cast<uint32_t>(2));
    ASSERT_EQ(FixedWidthInt(10, "0x2"), static_cast<uint16_t>(2));
    ASSERT_EQ(FixedWidthInt(10, "0x2"), static_cast<uint8_t>(2));
    ASSERT_EQ(FixedWidthInt(10, "0x2"), static_cast<int64_t>(2));
    ASSERT_EQ(FixedWidthInt(10, "0x2"), static_cast<int32_t>(2));
    ASSERT_EQ(FixedWidthInt(10, "0x2"), static_cast<int16_t>(2));
    ASSERT_EQ(FixedWidthInt(10, "0x2"), static_cast<int8_t>(2));
}

TEST(FixedWidthInt, cmp) {
    auto rng = INDEPENDENT_TEST_RNG();
    FixedWidthInt w(100);
    while (true) {
        w.randomize(rng);
        if (w.non_zero()) {
            break;
        }
    }
    auto w2 = w;
    ASSERT_FALSE(w2 < w);
    ASSERT_FALSE(w2 > w);
    ASSERT_TRUE(w2 <= w);
    ASSERT_TRUE(w2 >= w);
    ASSERT_TRUE(w2 == w);
    w2.decrement();
    ASSERT_TRUE(w2 < w);
    ASSERT_FALSE(w2 > w);
    ASSERT_TRUE(w2 <= w);
    ASSERT_FALSE(w2 >= w);
    ASSERT_FALSE(w2 == w);

    FixedWidthInt a(200, -1);
    FixedWidthInt b(100, -1);
    ASSERT_TRUE(a > b);
    ASSERT_TRUE(a >= b);
    ASSERT_FALSE(a == b);
    ASSERT_FALSE(a < b);
    ASSERT_FALSE(a <= b);
    ASSERT_FALSE(b > a);
    ASSERT_FALSE(b >= a);
    ASSERT_FALSE(b == a);
    ASSERT_TRUE(b < a);
    ASSERT_TRUE(b <= a);

    FixedWidthInt c(200, 0);
    FixedWidthInt d(100, 0);
    ASSERT_FALSE(c > d);
    ASSERT_TRUE(c >= d);
    ASSERT_TRUE(c == d);
    ASSERT_FALSE(c < d);
    ASSERT_TRUE(c <= d);
    ASSERT_FALSE(d > c);
    ASSERT_TRUE(d >= c);
    ASSERT_TRUE(d == c);
    ASSERT_FALSE(d < c);
    ASSERT_TRUE(d <= c);
}

TEST(FixedWidthInt, cmp_int) {
    FixedWidthInt big(100);
    big.increment(64);
    for (int64_t k1 = 0; k1 < 5; k1++) {
        for (int64_t k2 = -5; k2 < 5; k2++) {
            EXPECT_EQ(FixedWidthInt(0, 0) < static_cast<int64_t>(k2), 0 < k2) << k2;
            EXPECT_EQ(FixedWidthInt(0, 0) <= static_cast<int64_t>(k2), 0 <= k2);

            EXPECT_EQ(FixedWidthInt(10, k1) < static_cast<int64_t>(k2), k1 < k2);
            EXPECT_EQ(FixedWidthInt(10, k1) < static_cast<int32_t>(k2), k1 < k2);
            EXPECT_EQ(FixedWidthInt(10, k1) < static_cast<int16_t>(k2), k1 < k2);
            EXPECT_EQ(FixedWidthInt(10, k1) < static_cast<int8_t>(k2), k1 < k2);

            EXPECT_EQ(FixedWidthInt(10, k1) <= static_cast<int64_t>(k2), k1 <= k2);
            EXPECT_EQ(FixedWidthInt(10, k1) <= static_cast<int32_t>(k2), k1 <= k2);
            EXPECT_EQ(FixedWidthInt(10, k1) <= static_cast<int16_t>(k2), k1 <= k2);
            EXPECT_EQ(FixedWidthInt(10, k1) <= static_cast<int8_t>(k2), k1 <= k2);

            EXPECT_EQ(FixedWidthInt(100, k1) < static_cast<int64_t>(k2), k1 < k2);
            EXPECT_EQ(FixedWidthInt(100, k1) < static_cast<int32_t>(k2), k1 < k2);
            EXPECT_EQ(FixedWidthInt(100, k1) < static_cast<int16_t>(k2), k1 < k2);
            EXPECT_EQ(FixedWidthInt(100, k1) < static_cast<int8_t>(k2), k1 < k2);

            EXPECT_EQ(FixedWidthInt(100, k1) <= static_cast<int64_t>(k2), k1 <= k2);
            EXPECT_EQ(FixedWidthInt(100, k1) <= static_cast<int32_t>(k2), k1 <= k2);
            EXPECT_EQ(FixedWidthInt(100, k1) <= static_cast<int16_t>(k2), k1 <= k2);
            EXPECT_EQ(FixedWidthInt(100, k1) <= static_cast<int8_t>(k2), k1 <= k2);

            EXPECT_EQ(big < static_cast<int64_t>(k2), false);
            EXPECT_EQ(big < static_cast<int32_t>(k2), false);
            EXPECT_EQ(big < static_cast<int16_t>(k2), false);
            EXPECT_EQ(big < static_cast<int8_t>(k2), false);

            EXPECT_EQ(big <= static_cast<int64_t>(k2), false);
            EXPECT_EQ(big <= static_cast<int32_t>(k2), false);
            EXPECT_EQ(big <= static_cast<int16_t>(k2), false);
            EXPECT_EQ(big <= static_cast<int8_t>(k2), false);
        }
        for (int64_t k2 = 0; k2 < 5; k2++) {
            EXPECT_EQ(FixedWidthInt(10, k1) < static_cast<uint64_t>(k2), k1 < k2);
            EXPECT_EQ(FixedWidthInt(10, k1) < static_cast<uint32_t>(k2), k1 < k2);
            EXPECT_EQ(FixedWidthInt(10, k1) < static_cast<uint16_t>(k2), k1 < k2);
            EXPECT_EQ(FixedWidthInt(10, k1) < static_cast<uint8_t>(k2), k1 < k2);

            EXPECT_EQ(FixedWidthInt(10, k1) <= static_cast<uint64_t>(k2), k1 <= k2);
            EXPECT_EQ(FixedWidthInt(10, k1) <= static_cast<uint32_t>(k2), k1 <= k2);
            EXPECT_EQ(FixedWidthInt(10, k1) <= static_cast<uint16_t>(k2), k1 <= k2);
            EXPECT_EQ(FixedWidthInt(10, k1) <= static_cast<uint8_t>(k2), k1 <= k2);

            EXPECT_EQ(FixedWidthInt(100, k1) < static_cast<uint64_t>(k2), k1 < k2);
            EXPECT_EQ(FixedWidthInt(100, k1) < static_cast<uint32_t>(k2), k1 < k2);
            EXPECT_EQ(FixedWidthInt(100, k1) < static_cast<uint16_t>(k2), k1 < k2);
            EXPECT_EQ(FixedWidthInt(100, k1) < static_cast<uint8_t>(k2), k1 < k2);

            EXPECT_EQ(FixedWidthInt(100, k1) <= static_cast<uint64_t>(k2), k1 <= k2);
            EXPECT_EQ(FixedWidthInt(100, k1) <= static_cast<uint32_t>(k2), k1 <= k2);
            EXPECT_EQ(FixedWidthInt(100, k1) <= static_cast<uint16_t>(k2), k1 <= k2);
            EXPECT_EQ(FixedWidthInt(100, k1) <= static_cast<uint8_t>(k2), k1 <= k2);

            EXPECT_EQ(big < static_cast<uint64_t>(k2), false);
            EXPECT_EQ(big < static_cast<uint32_t>(k2), false);
            EXPECT_EQ(big < static_cast<uint16_t>(k2), false);
            EXPECT_EQ(big < static_cast<uint8_t>(k2), false);

            EXPECT_EQ(big <= static_cast<uint64_t>(k2), false);
            EXPECT_EQ(big <= static_cast<uint32_t>(k2), false);
            EXPECT_EQ(big <= static_cast<uint16_t>(k2), false);
            EXPECT_EQ(big <= static_cast<uint8_t>(k2), false);
        }
    }
}

TEST(FixedWidthInt, add_fuzz) {
    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t n1 = 0; n1 < 500; n1 += 100) {
        for (size_t n2 = 0; n2 < 500; n2 += 100) {
            FixedWidthInt actual = FixedWidthInt::random(n1, rng);
            FixedWidthInt offset = FixedWidthInt::random(n2, rng);

            FixedWidthInt expected = actual;
            for (size_t k = 0; k < n2; k++) {
                if (offset[k]) {
                    expected.increment(k);
                }
            }
            actual += offset;
            ASSERT_EQ(actual, expected);
        }
    }
}

TEST(FixedWidthInt, add_int_fuzz) {
    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t n1 = 0; n1 < 500; n1 += 100) {
        FixedWidthInt expected = FixedWidthInt::random(n1, rng);
        FixedWidthInt actual = expected;

        uint64_t offset_u64 = rng();
        actual += offset_u64;
        actual -= FixedWidthInt(n1, offset_u64);
        ASSERT_EQ(actual, expected);

        int64_t offset_i64 = (int64_t)(offset_u64 & ~(-1ULL << 63));
        if (offset_u64 >> 63) {
            offset_i64 += INT64_MIN;
        }
        actual += offset_i64;
        actual -= FixedWidthInt(n1, offset_i64);
        ASSERT_EQ(actual, expected);
    }
}

TEST(FixedWidthInt, sub_fuzz) {
    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t n1 = 0; n1 < 500; n1 += 100) {
        for (size_t n2 = 0; n2 < 500; n2 += 100) {
            FixedWidthInt actual = FixedWidthInt::random(n1, rng);
            FixedWidthInt offset = FixedWidthInt::random(n2, rng);

            FixedWidthInt expected = actual;
            for (size_t k = 0; k < n2; k++) {
                if (offset[k]) {
                    expected.decrement(k);
                }
            }
            actual -= offset;
            ASSERT_EQ(actual, expected);
        }
    }
}

TEST(FixedWidthInt, sub_int_fuzz) {
    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t n1 = 0; n1 < 500; n1 += 100) {
        FixedWidthInt expected = FixedWidthInt::random(n1, rng);
        FixedWidthInt actual = expected;

        uint64_t offset_u64 = rng();
        actual -= offset_u64;
        actual += FixedWidthInt(n1, offset_u64);
        ASSERT_EQ(actual, expected);

        int64_t offset_i64 = static_cast<int64_t>(offset_u64 & ~(-1ULL << 63));
        if (offset_u64 >> 63) {
            offset_i64 += INT64_MIN;
        }
        actual -= offset_i64;
        actual += FixedWidthInt(n1, offset_i64);
        ASSERT_EQ(actual, expected);
    }
}

TEST(FixedWidthInt, shift_fuzz) {
    auto rng = INDEPENDENT_TEST_RNG();
    std::vector<int64_t> shifts{
        -500, -129, -128, -127, -100, -64, -50, -32, -16, -1, 0, 1, 16, 32, 50, 64, 100, 127, 128, 129, 500,
    };
    for (auto shift : shifts) {
        FixedWidthInt start = FixedWidthInt::random(300, rng);
        FixedWidthInt actual = start;

        actual >>= shift;
        FixedWidthInt expected(300);
        for (int64_t k = 0; k < 300; k++) {
            int64_t k2 = k + shift;
            if (0 <= k2 && start[static_cast<size_t>(k2)]) {
                expected.increment(k);
            }
        }
        EXPECT_EQ(actual, expected) << shift;
    }
}

TEST(FixedWidthInt, signed_shift) {
    auto rng = INDEPENDENT_TEST_RNG();
    FixedWidthInt start = FixedWidthInt::random(300, rng);
    start.set_bit(299);
    start.isigned_right_shift(300);
    start ^= -1;
    ASSERT_FALSE(start.non_zero());
}

TEST(FixedWidthInt, idouble_mod) {
    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t n = 2; n < 50; n++) {
        FixedWidthInt mod = FixedWidthInt::random(n, rng);
        mod.set_bit(n - 1);
        FixedWidthInt v = FixedWidthInt::random_mod(rng, mod);
        FixedWidthInt v_old = v;
        v.idouble_mod(mod);
        ASSERT_EQ(v, (v_old << 1) % mod);
    }
}

TEST(FixedWidthInt, ihalve_mod) {
    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t n = 2; n < 50; n++) {
        FixedWidthInt mod = FixedWidthInt::random(n, rng);
        mod.set_bit(n - 1);
        mod.set_bit(0);
        FixedWidthInt v = FixedWidthInt::random_mod(rng, mod);
        FixedWidthInt v_old = v;
        v.ihalve_mod(mod);
        ASSERT_EQ(v_old, (v << 1) % mod);
    }
}

TEST(FixedWidthInt, read_bit_shifted_word_fuzz) {
    auto rng = INDEPENDENT_TEST_RNG();
    FixedWidthInt v = FixedWidthInt::random(150, rng);
    for (int64_t offset = -100; offset < 200; offset++) {
        uint64_t actual = v.read_bit_shifted_word(offset);
        uint64_t expected = 0;
        for (int64_t k = 0; k < 64; k++) {
            if (offset + k >= 0 && v[static_cast<uint64_t>(offset + k)]) {
                expected ^= static_cast<uint64_t>(1) << k;
            }
        }
        ASSERT_EQ(actual, expected);
    }
}

TEST(FixedWidthInt, mul_fuzz) {
    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t n1 = 0; n1 < 500; n1 += 100) {
        for (size_t n2 = 0; n2 < 500; n2 += 100) {
            FixedWidthInt actual = FixedWidthInt::random(n1, rng);
            FixedWidthInt value = FixedWidthInt::random(n2, rng);

            FixedWidthInt expected(n1);
            for (size_t k1 = 0; k1 < n1; k1++) {
                for (size_t k2 = 0; k2 < n2; k2++) {
                    if (actual[k1] && value[k2]) {
                        expected.increment(k1 + k2);
                    }
                }
            }
            actual *= value;
            ASSERT_EQ(actual, expected) << n1 << ", " << n2;
        }
    }
}

TEST(FixedWidthInt, iadd_product) {
    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t n1 = 100; n1 < 200; n1 += 31) {
        for (size_t n2 = 100; n2 < 200; n2 += 29) {
            for (size_t n3 = 100; n3 < 200; n3 += 33) {
                FixedWidthInt v1 = FixedWidthInt::random(n1, rng);
                FixedWidthInt v2 = FixedWidthInt::random(n2, rng);
                FixedWidthInt v3 = FixedWidthInt::random(n3, rng);
                FixedWidthInt expected = v3;
                for (size_t k1 = 0; k1 < v1.num_bits; k1++) {
                    if (v1[k1]) {
                        expected.iadd_shifted(v2, k1);
                    }
                }
                v3.iadd_product(v1, v2);
                ASSERT_EQ(v3, expected);
            }
        }
    }
}

TEST(FixedWidthInt, iadd_product64) {
    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t n1 = 100; n1 < 200; n1 += 31) {
        for (size_t n3 = 100; n3 < 200; n3 += 33) {
            FixedWidthInt v1 = FixedWidthInt::random(n1, rng);
            uint64_t v2 = rng();
            FixedWidthInt v3 = FixedWidthInt::random(n3, rng);
            FixedWidthInt expected = v3;
            for (size_t k2 = 0; k2 < 64; k2++) {
                if ((v2 >> k2) & 1) {
                    expected.iadd_shifted(v1, k2);
                }
            }
            v3.iadd_product(v1, v2);
            ASSERT_EQ(v3, expected);
        }
    }
}

TEST(FixedWidthInt, iadd_product_shifted) {
    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t n1 = 100; n1 < 200; n1 += 31) {
        for (size_t n3 = 100; n3 < 200; n3 += 33) {
            for (size_t shift = 0; shift < 100; shift += 27) {
                FixedWidthInt v1 = FixedWidthInt::random(n1, rng);
                uint64_t v2 = rng();
                FixedWidthInt v3 = FixedWidthInt::random(n3, rng);
                FixedWidthInt expected = v3;
                for (size_t k2 = 0; k2 < 64; k2++) {
                    if ((v2 >> k2) & 1) {
                        expected.iadd_shifted(v1, k2 + shift);
                    }
                }
                v3.iadd_product_shifted(v1, v2, shift);
                ASSERT_EQ(v3, expected) << n1 << ", " << n3 << ", " << shift;
            }
        }
    }
}

TEST(FixedWidthInt, mul64_fuzz) {
    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t n1 = 0; n1 < 500; n1 += 100) {
        FixedWidthInt actual = FixedWidthInt::random(n1, rng);
        auto value = rng();

        FixedWidthInt expected(n1);
        for (size_t k = 0; k < 64; k++) {
            if ((value >> k) & 1) {
                expected.iadd_shifted(actual, k);
            }
        }
        actual *= value;
        ASSERT_EQ(actual, expected) << n1;
    }
}

TEST(FixedWidthInt, add_shifted_fuzz) {
    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t n1 = 0; n1 < 500; n1 += 100) {
        for (size_t n2 = 0; n2 < 500; n2 += 100) {
            for (auto shift : std::vector<size_t>{0, 1, 2, 16, 63, 64, 65, 127, 128, 19}) {
                FixedWidthInt actual = FixedWidthInt::random(n1, rng);
                FixedWidthInt value = FixedWidthInt::random(n2, rng);

                FixedWidthInt expected = actual;
                for (size_t k2 = 0; k2 < n2; k2++) {
                    if (value[k2]) {
                        actual.increment(k2 + shift);
                    }
                }
                expected.iadd_shifted(value, shift);
                ASSERT_EQ(actual, expected) << n1 << ", " << n2 << ", " << shift;
            }
        }
    }
}

TEST(randomize_mod, covers) {
    auto rng = INDEPENDENT_TEST_RNG();
    FixedWidthInt modulus(254);
    FixedWidthInt total(254);
    modulus.randomize(rng);
    modulus.words[3] = 0;
    modulus.words[2] &= 0xFFFFFF;
    FixedWidthInt v(195);
    for (size_t k = 0; k < (1 << 13); k++) {
        v.randomize_mod(rng, modulus);
        ASSERT_LT(v, modulus);
        total += v;
    }
    ASSERT_GT(total, modulus << 11);
    ASSERT_GT(modulus << 13, total);
}

TEST(fixed_width_int, num_bits_in_use) {
    ASSERT_EQ(FixedWidthInt(0, 0).num_bits_in_use(), 0);
    ASSERT_EQ(FixedWidthInt(1, 1).num_bits_in_use(), 1);
    ASSERT_EQ(FixedWidthInt(1, 0).num_bits_in_use(), 0);
    ASSERT_EQ(FixedWidthInt(100, -1).num_bits_in_use(), 100);
    ASSERT_EQ(FixedWidthInt(65, -1).num_bits_in_use(), 65);
    ASSERT_EQ(FixedWidthInt(64, -1).num_bits_in_use(), 64);
    ASSERT_EQ(FixedWidthInt(63, -1).num_bits_in_use(), 63);
    ASSERT_EQ(FixedWidthInt(63, 0x1F).num_bits_in_use(), 5);
}

TEST(FixedWidthInt, remainder) {
    auto val = FixedWidthInt(
        500,
        "0x894363a66c49c1dc9d9e556cfae4cef6a356c832ab31050ba9563a9867a6f37aa03ec9428c510676acd35e4d55ff44830b27257d68d5"
        "b65a09244423d57d0");
    auto mod = FixedWidthInt(100, "0x75bb303d8222019b44e46708a");
    auto expected = FixedWidthInt(100, "0x1f1c7f606bfa7649a21431af8");
    auto actual = val % mod;
    ASSERT_EQ(actual.num_bits, 100);
    ASSERT_EQ(actual, expected);

    ASSERT_EQ(val % FixedWidthInt(10, 1), 0);
    ASSERT_EQ(val % FixedWidthInt(10, 2), val[0]);
    ASSERT_EQ(mod % val, mod);

    EXPECT_THROW({ val % FixedWidthInt(10, 0); }, std::invalid_argument);

    val = FixedWidthInt(
        500,
        "0xe5b5cb3471346875dc35ce88274cde703d99e738d393c16a99ca4baddf20a2ce6ccaec3f7f6a89c33126fbdb4e0cf0f9fe66db7c0341"
        "ccca9a50fe9ff0230");
    mod = FixedWidthInt(
        400, "0x638e2a6d69f93a00e6bd4a405460febba07267a600e85e4f2455cfc3f9301349474c3ce123c68f948c3b1435edb297f7c02");
    ASSERT_EQ(
        val % mod,
        FixedWidthInt(
            400,
            "0x3b62d65c063c6b9248f7888be5c0adada4bee350d4f7cc09d39da2538782c33bfcb459e65f77e3804c825ceb3f352efd86e"));
}

TEST(FixedWidthInt, iadd_carry) {
    FixedWidthInt f(20, -1);
    ASSERT_TRUE(f.iadd_carry(FixedWidthInt(1, 1)));

    f = FixedWidthInt(20, -2);
    ASSERT_FALSE(f.iadd_carry(FixedWidthInt(1, 1)));

    f = FixedWidthInt(200, -1);
    ASSERT_TRUE(f.iadd_carry(FixedWidthInt(1, 1)));

    f = FixedWidthInt(200, -2);
    ASSERT_FALSE(f.iadd_carry(FixedWidthInt(1, 1)));

    f = FixedWidthInt(100, -10);
    ASSERT_FALSE(f.iadd_carry(FixedWidthInt(100, 9)));

    f = FixedWidthInt(100, -10);
    ASSERT_TRUE(f.iadd_carry(FixedWidthInt(100, 10)));
}

TEST(FixedWidthInt, isub_borrow) {
    FixedWidthInt f(20, 0);
    ASSERT_TRUE(f.isub_borrow(FixedWidthInt(1, 1)));

    f = FixedWidthInt(20, 1);
    ASSERT_FALSE(f.isub_borrow(FixedWidthInt(1, 1)));

    f = FixedWidthInt(200, 0);
    ASSERT_TRUE(f.isub_borrow(FixedWidthInt(1, 1)));

    f = FixedWidthInt(200, 1);
    ASSERT_FALSE(f.isub_borrow(FixedWidthInt(1, 1)));

    f = FixedWidthInt(100, 10);
    ASSERT_FALSE(f.isub_borrow(FixedWidthInt(100, 10)));

    f = FixedWidthInt(100, 10);
    ASSERT_TRUE(f.isub_borrow(FixedWidthInt(100, 11)));
}

TEST(FixedWidthInt, iadd_mod) {
    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t n = 2; n < 100; n += 10) {
        auto mod = FixedWidthInt::random(n, rng);
        if (!mod) {
            mod += 1;
        }
        auto off = FixedWidthInt::random(n, rng);
        FixedWidthInt val(n);
        val.randomize_mod(rng, mod);

        auto old_val = val;
        val.iadd_mod(off, mod);
        ASSERT_LT(val, mod);
        auto expected = (old_val + off) % mod;
        if (val != expected) {
            ASSERT_EQ(val, expected) << "\n    val=" << old_val << "\n    off=" << off << "\n    mod=" << mod
                                     << "\n    act=" << val << "\n    exp=" << expected;
        }
    }
}

TEST(FixedWidthInt, isub_mod) {
    auto rng = INDEPENDENT_TEST_RNG();
    for (size_t n = 2; n < 100; n += 10) {
        auto mod = FixedWidthInt::random(n, rng);
        if (!mod) {
            mod += 1;
        }
        auto off = FixedWidthInt::random(n, rng);
        FixedWidthInt val(n);
        val.randomize_mod(rng, mod);

        auto old_val = val;
        val.isub_mod(off, mod);
        ASSERT_LT(val, mod);
        auto restored = (val + off) % mod;
        if (old_val != restored) {
            ASSERT_EQ(old_val, restored) << "\n    val=" << old_val << "\n    off=" << off << "\n    mod=" << mod
                                         << "\n    act=" << val << "\n    res=" << restored;
        }
    }
}

TEST(FixedWidthInt, is_coprime_to) {
    auto f = [](uint64_t v) {
        return FixedWidthInt(10, v);
    };
    ASSERT_EQ(f(0).is_coprime_to(f(0)), false);
    ASSERT_EQ(f(1).is_coprime_to(f(0)), true);
    ASSERT_EQ(f(2).is_coprime_to(f(0)), false);
    ASSERT_EQ(f(3).is_coprime_to(f(0)), false);
    ASSERT_EQ(f(4).is_coprime_to(f(0)), false);
    ASSERT_EQ(f(5).is_coprime_to(f(0)), false);
    ASSERT_EQ(f(6).is_coprime_to(f(0)), false);

    ASSERT_EQ(f(0).is_coprime_to(f(1)), true);
    ASSERT_EQ(f(1).is_coprime_to(f(1)), true);
    ASSERT_EQ(f(2).is_coprime_to(f(1)), true);
    ASSERT_EQ(f(3).is_coprime_to(f(1)), true);
    ASSERT_EQ(f(4).is_coprime_to(f(1)), true);
    ASSERT_EQ(f(5).is_coprime_to(f(1)), true);
    ASSERT_EQ(f(6).is_coprime_to(f(1)), true);

    ASSERT_EQ(f(0).is_coprime_to(f(2)), false);
    ASSERT_EQ(f(1).is_coprime_to(f(2)), true);
    ASSERT_EQ(f(2).is_coprime_to(f(2)), false);
    ASSERT_EQ(f(3).is_coprime_to(f(2)), true);
    ASSERT_EQ(f(4).is_coprime_to(f(2)), false);
    ASSERT_EQ(f(5).is_coprime_to(f(2)), true);
    ASSERT_EQ(f(6).is_coprime_to(f(2)), false);

    ASSERT_EQ(f(0).is_coprime_to(f(3)), false);
    ASSERT_EQ(f(1).is_coprime_to(f(3)), true);
    ASSERT_EQ(f(2).is_coprime_to(f(3)), true);
    ASSERT_EQ(f(3).is_coprime_to(f(3)), false);
    ASSERT_EQ(f(4).is_coprime_to(f(3)), true);
    ASSERT_EQ(f(5).is_coprime_to(f(3)), true);
    ASSERT_EQ(f(6).is_coprime_to(f(3)), false);

    ASSERT_EQ(f(0).is_coprime_to(f(4)), false);
    ASSERT_EQ(f(1).is_coprime_to(f(4)), true);
    ASSERT_EQ(f(2).is_coprime_to(f(4)), false);
    ASSERT_EQ(f(3).is_coprime_to(f(4)), true);
    ASSERT_EQ(f(4).is_coprime_to(f(4)), false);
    ASSERT_EQ(f(5).is_coprime_to(f(4)), true);
    ASSERT_EQ(f(6).is_coprime_to(f(4)), false);

    ASSERT_EQ(f(0).is_coprime_to(f(5)), false);
    ASSERT_EQ(f(1).is_coprime_to(f(5)), true);
    ASSERT_EQ(f(2).is_coprime_to(f(5)), true);
    ASSERT_EQ(f(3).is_coprime_to(f(5)), true);
    ASSERT_EQ(f(4).is_coprime_to(f(5)), true);
    ASSERT_EQ(f(5).is_coprime_to(f(5)), false);
    ASSERT_EQ(f(6).is_coprime_to(f(5)), true);

    ASSERT_EQ(f(0).is_coprime_to(f(6)), false);
    ASSERT_EQ(f(1).is_coprime_to(f(6)), true);
    ASSERT_EQ(f(2).is_coprime_to(f(6)), false);
    ASSERT_EQ(f(3).is_coprime_to(f(6)), false);
    ASSERT_EQ(f(4).is_coprime_to(f(6)), false);
    ASSERT_EQ(f(5).is_coprime_to(f(6)), true);
    ASSERT_EQ(f(6).is_coprime_to(f(6)), false);

    auto a = FixedWidthInt(200, "0xb1634a6dcbf5fd6edf20f1d258cd06afbcc7a9156ff8949de6");
    auto b = FixedWidthInt(200, "0x2ba40a43bd9893e6bd683f6ed4ddc1f6f71b4c27548ade992d");
    ASSERT_TRUE(a.is_coprime_to(b));

    a = FixedWidthInt(205, "0xb1634a6dcbf5fd6edf20f1d258cd06afbcc7a9156ff8949de6");
    b = FixedWidthInt(205, "0x2ba40a43bd9893e6bd683f6ed4ddc1f6f71b4c27548ade992d");
    ASSERT_TRUE(a.is_coprime_to(b));

    a *= f(7);
    b *= f(7);
    ASSERT_FALSE(a.is_coprime_to(b));
}

TEST(FixedWidthInt, inplace_times5) {
    FixedWidthInt f(200, 13);

    ASSERT_FALSE(f.inplace_times5());
    ASSERT_EQ(f, FixedWidthInt(200, 13 * 5));
    f.inplace_div5();
    ASSERT_EQ(f, FixedWidthInt(200, 13));

    f = FixedWidthInt(2, "1");
    ASSERT_TRUE(f.inplace_times5());
    ASSERT_EQ(f, 1);

    f = FixedWidthInt(3, "1");
    ASSERT_FALSE(f.inplace_times5());
    ASSERT_EQ(f, 5);
    ASSERT_TRUE(f.inplace_times5());
    ASSERT_EQ(f, 1);

    f = FixedWidthInt(4, "1");
    ASSERT_FALSE(f.inplace_times5());
    ASSERT_EQ(f, 5);
    ASSERT_TRUE(f.inplace_times5());
    ASSERT_EQ(f, 25 % 16);

    f = FixedWidthInt(5, "1");
    ASSERT_FALSE(f.inplace_times5());
    ASSERT_EQ(f, 5);
    ASSERT_FALSE(f.inplace_times5());
    ASSERT_EQ(f, 25);
    ASSERT_TRUE(f.inplace_times5());
    ASSERT_EQ(f, 125 % 32);

    f = FixedWidthInt(200, "0x1923e29469a58c555efb036f4195c2e9d5e0d9ab97c0d8ec59");
    ASSERT_FALSE(f.inplace_times5());
    ASSERT_EQ(f, FixedWidthInt(200, "0x7db36ce6103bbdaadae7112c47ecce912d644059f6c43c9dbd"));
    f.inplace_div5();
    ASSERT_EQ(f, FixedWidthInt(200, "0x1923e29469a58c555efb036f4195c2e9d5e0d9ab97c0d8ec59"));
    ASSERT_FALSE(f.inplace_times5());
    ASSERT_EQ(f, FixedWidthInt(200, "0x7db36ce6103bbdaadae7112c47ecce912d644059f6c43c9dbd"));
    ASSERT_TRUE(f.inplace_times5());
    ASSERT_EQ(f, FixedWidthInt(200, "0x7481207e512ab456468355dd67a008d5e2f541c1d1d52f14b1"));

    f = FixedWidthInt(5, "6");
    f.inplace_div5();
    ASSERT_EQ(f, 1);

    f = FixedWidthInt(70, "0x50000000000000000");
    f.inplace_div5();
    ASSERT_EQ(f, FixedWidthInt(70, "0x10000000000000000"));

    f = FixedWidthInt(70, "0x50000000000000001");
    f.inplace_div5();
    ASSERT_EQ(f, FixedWidthInt(70, "0x10000000000000000"));
}

TEST(FixedWidthInt, decimal) {
    ASSERT_EQ(FixedWidthInt(100, "0").decimal(), "0");
    ASSERT_EQ(FixedWidthInt(100, "1").decimal(), "1");
    ASSERT_EQ(FixedWidthInt(100, "2").decimal(), "2");
    ASSERT_EQ(FixedWidthInt(100, "3").decimal(), "3");
    ASSERT_EQ(FixedWidthInt(100, "4").decimal(), "4");
    ASSERT_EQ(FixedWidthInt(100, "5").decimal(), "5");
    ASSERT_EQ(FixedWidthInt(100, "6").decimal(), "6");
    ASSERT_EQ(FixedWidthInt(100, "7").decimal(), "7");
    ASSERT_EQ(FixedWidthInt(100, "8").decimal(), "8");
    ASSERT_EQ(FixedWidthInt(100, "9").decimal(), "9");
    ASSERT_EQ(FixedWidthInt(100, "10").decimal(), "10");
    ASSERT_EQ(FixedWidthInt(100, "11").decimal(), "11");
    ASSERT_EQ(FixedWidthInt(100, "12").decimal(), "12");
    ASSERT_EQ(FixedWidthInt(100, "13").decimal(), "13");
    ASSERT_EQ(FixedWidthInt(100, "14").decimal(), "14");
    ASSERT_EQ(FixedWidthInt(100, "15").decimal(), "15");

    ASSERT_EQ(
        FixedWidthInt(300, "927691359788317178299769504960248623287550024633243471194917569428669197023004368826224202")
            .decimal(),
        "927691359788317178299769504960248623287550024633243471194917569428669197023004368826224202");
}

TEST(FixedWidthInt, from_str_hex) {
    ASSERT_THROW({ FixedWidthInt::from_str_hex("0x0F", 4); }, std::invalid_argument);
    ASSERT_THROW({ FixedWidthInt::from_str_hex("0x1F", 4); }, std::invalid_argument);
    ASSERT_THROW({ FixedWidthInt::from_str_hex("0x2F", 5); }, std::invalid_argument);
    ASSERT_THROW({ FixedWidthInt::from_str_hex("0x3F", 5); }, std::invalid_argument);
    ASSERT_THROW({ FixedWidthInt::from_str_hex("0x4F", 6); }, std::invalid_argument);
    ASSERT_THROW({ FixedWidthInt::from_str_hex("0x5F", 6); }, std::invalid_argument);
    ASSERT_THROW({ FixedWidthInt::from_str_hex("0x6F", 6); }, std::invalid_argument);
    ASSERT_THROW({ FixedWidthInt::from_str_hex("0x7F", 6); }, std::invalid_argument);
    ASSERT_THROW({ FixedWidthInt::from_str_hex("0x8F", 7); }, std::invalid_argument);
    ASSERT_THROW({ FixedWidthInt::from_str_hex("0x9F", 7); }, std::invalid_argument);
    ASSERT_THROW({ FixedWidthInt::from_str_hex("0xaF", 7); }, std::invalid_argument);
    ASSERT_THROW({ FixedWidthInt::from_str_hex("0xbF", 7); }, std::invalid_argument);
    ASSERT_THROW({ FixedWidthInt::from_str_hex("0xcF", 7); }, std::invalid_argument);
    ASSERT_THROW({ FixedWidthInt::from_str_hex("0xdF", 7); }, std::invalid_argument);
    ASSERT_THROW({ FixedWidthInt::from_str_hex("0xeF", 7); }, std::invalid_argument);
    ASSERT_THROW({ FixedWidthInt::from_str_hex("0xfF", 7); }, std::invalid_argument);
    ASSERT_THROW({ FixedWidthInt::from_str_hex("0xAF", 7); }, std::invalid_argument);
    ASSERT_THROW({ FixedWidthInt::from_str_hex("0xBF", 7); }, std::invalid_argument);
    ASSERT_THROW({ FixedWidthInt::from_str_hex("0xCF", 7); }, std::invalid_argument);
    ASSERT_THROW({ FixedWidthInt::from_str_hex("0xDF", 7); }, std::invalid_argument);
    ASSERT_THROW({ FixedWidthInt::from_str_hex("0xEF", 7); }, std::invalid_argument);
    ASSERT_THROW({ FixedWidthInt::from_str_hex("0xFF", 7); }, std::invalid_argument);

    ASSERT_EQ(FixedWidthInt::from_str_hex("0x0F", 5), 0x0F);
    ASSERT_EQ(FixedWidthInt::from_str_hex("0x1F", 5), 0x1F);
    ASSERT_EQ(FixedWidthInt::from_str_hex("0x2F", 6), 0x2F);
    ASSERT_EQ(FixedWidthInt::from_str_hex("0x3F", 6), 0x3F);
    ASSERT_EQ(FixedWidthInt::from_str_hex("0x4F", 7), 0x4F);
    ASSERT_EQ(FixedWidthInt::from_str_hex("0x5F", 7), 0x5F);
    ASSERT_EQ(FixedWidthInt::from_str_hex("0x6F", 7), 0x6F);
    ASSERT_EQ(FixedWidthInt::from_str_hex("0x7F", 7), 0x7F);
    ASSERT_EQ(FixedWidthInt::from_str_hex("0x8F", 8), 0x8F);
    ASSERT_EQ(FixedWidthInt::from_str_hex("0x9F", 8), 0x9F);
    ASSERT_EQ(FixedWidthInt::from_str_hex("0xaF", 8), 0xaF);
    ASSERT_EQ(FixedWidthInt::from_str_hex("0xbF", 8), 0xbF);
    ASSERT_EQ(FixedWidthInt::from_str_hex("0xcF", 8), 0xcF);
    ASSERT_EQ(FixedWidthInt::from_str_hex("0xdF", 8), 0xdF);
    ASSERT_EQ(FixedWidthInt::from_str_hex("0xeF", 8), 0xeF);
    ASSERT_EQ(FixedWidthInt::from_str_hex("0xfF", 8), 0xfF);
    ASSERT_EQ(FixedWidthInt::from_str_hex("0xAF", 8), 0xAF);
    ASSERT_EQ(FixedWidthInt::from_str_hex("0xBF", 8), 0xBF);
    ASSERT_EQ(FixedWidthInt::from_str_hex("0xCF", 8), 0xCF);
    ASSERT_EQ(FixedWidthInt::from_str_hex("0xDF", 8), 0xDF);
    ASSERT_EQ(FixedWidthInt::from_str_hex("0xEF", 8), 0xEF);
    ASSERT_EQ(FixedWidthInt::from_str_hex("0xFF", 8), 0xFF);

    ASSERT_EQ(FixedWidthInt::from_str_hex("0x0F"), 0x0F);
    ASSERT_EQ(FixedWidthInt::from_str_hex("0xFF"), 0xFF);
    ASSERT_EQ(FixedWidthInt::from_str_hex("0x0F", 5).num_bits, 5);
    ASSERT_EQ(FixedWidthInt::from_str_hex("0x0F", 6).num_bits, 6);
    ASSERT_EQ(FixedWidthInt::from_str_hex("0x0F", 7).num_bits, 7);
    ASSERT_EQ(FixedWidthInt::from_str_hex("0x0F", 8).num_bits, 8);
    ASSERT_EQ(FixedWidthInt::from_str_hex("0xFF", 8).num_bits, 8);
}

TEST(FixedWithInt, mult_inverse) {
    ASSERT_EQ(FixedWidthInt("5").mult_inverse(FixedWidthInt("21")), 17);
    ASSERT_EQ(FixedWidthInt("50005").mult_inverse(FixedWidthInt("21")), 16);
    ASSERT_EQ(
        FixedWidthInt("0x83ff8b97878e1ef341be70b11").mult_inverse(FixedWidthInt("0x2f4d353a698aae1acb32f70c3")),
        FixedWidthInt("0x2c69118e93fdeb63da3959d08"));
}

TEST(FixedWidthInt, to_double) {
    ASSERT_EQ((double)FixedWidthInt("0"), 0.0);
    ASSERT_EQ((double)FixedWidthInt("1"), 1.0);
    ASSERT_EQ((double)FixedWidthInt("2"), 2.0);
    ASSERT_EQ((double)FixedWidthInt("3"), 3.0);
    ASSERT_EQ((double)FixedWidthInt("17"), 17.0);
    ASSERT_EQ((double)FixedWidthInt("0x123456789ABCD"), (double)0x123456789ABCDULL);
    ASSERT_EQ((double)FixedWidthInt("0xFFFFFFFFFFFFF"), (double)0xFFFFFFFFFFFFFULL);

    // Largest representable odd number.
    double m = (double)FixedWidthInt("0x1FFFFFFFFFFFFF");
    ASSERT_TRUE(m != (double)(0x1FFFFFFFFFFFFFULL - 1u));
    ASSERT_TRUE(m == (double)(0x1FFFFFFFFFFFFFULL + 0u));
    ASSERT_TRUE(m != (double)(0x1FFFFFFFFFFFFFULL + 1u));

    // Too big; rounded.
    double r = (double)FixedWidthInt("0x20000000000001");
    ASSERT_TRUE(r == (double)0x20000000000000 || r == (double)0x20000000000002);

    ASSERT_EQ((double)(FixedWidthInt("0x1FFFFFFFFFFFFF") << 511), 0x1FFFFFFFFFFFFFULL * pow(2, 511));
    ASSERT_EQ((double)(FixedWidthInt("0x1234567890ABCD") << 511), 0x1234567890ABCDULL * pow(2, 511));

    ASSERT_EQ((double)(FixedWidthInt("1") << 1022), pow(2, 1022));
    ASSERT_EQ((double)(FixedWidthInt("1") << 1023), pow(2, 1023));
    ASSERT_EQ(INFINITY, pow(2, 1024));
    ASSERT_THROW({ (double)(FixedWidthInt("1") << 1024); }, std::invalid_argument);
    ASSERT_THROW({ (double)(FixedWidthInt("1") << 3000); }, std::invalid_argument);
    ASSERT_EQ((double)(FixedWidthInt("0x1FFFFFFFFFFFFF_FFFFFFFFFF") << 511), 0x1FFFFFFFFFFFFFULL * pow(2, 551));
    ASSERT_EQ((double)(FixedWidthInt("0x1FFFFFFFFFFFFF_CCCCCCCCCC") << 511), 0x1FFFFFFFFFFFFFULL * pow(2, 551));
    ASSERT_EQ((double)(FixedWidthInt("0x1FFFFFFFFFFFFF_1111111111") << 511), 0x1FFFFFFFFFFFFFULL * pow(2, 551));
    ASSERT_EQ((double)(FixedWidthInt("0x1FFFFFFFFFFFFF_0000000000") << 511), 0x1FFFFFFFFFFFFFULL * pow(2, 551));
}

TEST(FixedWidthInt, to_approx_mantissa) {
    ASSERT_EQ(FixedWidthInt("0").to_approx_mantissa(), 0.0);
    ASSERT_EQ(FixedWidthInt("1").to_approx_mantissa(), 1.0);
    ASSERT_EQ(FixedWidthInt("2").to_approx_mantissa(), 1.0);
    ASSERT_EQ(FixedWidthInt("3").to_approx_mantissa(), 1.5);
    ASSERT_EQ(FixedWidthInt("4").to_approx_mantissa(), 1.0);
    ASSERT_EQ(FixedWidthInt("5").to_approx_mantissa(), 1.25);
    ASSERT_EQ(FixedWidthInt("6").to_approx_mantissa(), 1.5);
    ASSERT_EQ(FixedWidthInt("7").to_approx_mantissa(), 1.75);
    ASSERT_EQ(FixedWidthInt("8").to_approx_mantissa(), 1.0);

    auto m = FixedWidthInt("0x1FFFFFFFFFFFFF");
    ASSERT_EQ(m.to_approx_mantissa() * pow(2, m.num_bits_in_use() - 1), (double)m);
    m = FixedWidthInt("0x1234567890ABCD");
    ASSERT_EQ(m.to_approx_mantissa() * pow(2, m.num_bits_in_use() - 1), (double)m);
    m = FixedWidthInt("0x1234567890ABCE") << 763;
    ASSERT_EQ(m.to_approx_mantissa() * pow(2, m.num_bits_in_use() - 1), (double)m);
}
