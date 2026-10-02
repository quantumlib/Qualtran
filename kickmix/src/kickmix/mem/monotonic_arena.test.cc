#include "kickmix/mem/monotonic_arena.h"

#include "gtest/gtest.h"

using namespace kickmix;

TEST(grow_buf, allocation_alignment) {
    for (size_t alignment : {1, 2, 4, 8, 16, 32, 64}) {
        MonotonicArena_Helper buf;
        char *data = buf.grab_writeable(13, alignment);
        ASSERT_EQ(reinterpret_cast<uintptr_t>(data) % alignment, 0);
        data[12] = 7;

        char *grown = buf.grab_writeable(1000, alignment);
        ASSERT_EQ(reinterpret_cast<uintptr_t>(grown) % alignment, 0);
        ASSERT_EQ(data[12], 7);
        grown[999] = 9;
    }
}

TEST(grow_buf, grab_writeable__size__contiguous_copy_of_contents) {
    MonotonicArena<uint32_t, 4> buf;
    uint32_t *data = buf.grab_writeable(4);
    ASSERT_EQ(buf.size(), 4u);
    uint32_t *data2 = buf.grab_writeable(100);
    ASSERT_EQ(buf.size(), 104u);
    uint32_t *data3 = buf.grab_writeable(12);
    ASSERT_EQ(buf.size(), 116u);
    uint32_t *data4 = buf.grab_writeable(12);
    ASSERT_EQ(buf.size(), 128u);
    data[3] = 1;
    data2[2] = 4;
    data3[0] = 2;
    data4[11] = 11;

    uint32_t *all = new uint32_t[buf.size()];
    buf.memcpy_into(all);
    ASSERT_TRUE(((intptr_t)data & 0b1111) == 0);
    ASSERT_TRUE(((intptr_t)data2 & 0b1111) == 0);
    ASSERT_TRUE(((intptr_t)data3 & 0b1111) == 0);
    ASSERT_TRUE(((intptr_t)data4 & 0b1111) == 0);
    ASSERT_TRUE(((intptr_t)all & 0b1111) == 0);
    ASSERT_EQ(all[3], 1u);
    ASSERT_EQ(all[4 + 2], 4u);
    ASSERT_EQ(all[4 + 100], 2u);
    ASSERT_EQ(all[4 + 100 + 12 + 11], 11u);
    delete[] all;
}

TEST(grow_buf, push_back__iter_spans__clear) {
    MonotonicArena<uint16_t, 4> buf;
    buf.push_back(3);
    buf.push_back_many(std::vector<uint16_t>{4, 5, 6, 7});
    buf.push_back_many_reversed(std::vector<uint16_t>{10, 9, 8});
    buf.push_back_repeat(2, 5);

    std::vector<uint16_t> out;
    buf.iter_spans([&](std::span<uint16_t> data) {
        out.insert(out.end(), data.begin(), data.end());
    });
    ASSERT_EQ(out, (std::vector<uint16_t>{3, 4, 5, 6, 7, 8, 9, 10, 2, 2, 2, 2, 2}));

    buf.clear();
    out.clear();
    buf.iter_spans([&](std::span<uint16_t> data) {
        out.insert(out.end(), data.begin(), data.end());
    });
    ASSERT_EQ(out, (std::vector<uint16_t>{}));

    buf.push_back(3);
    out.clear();
    buf.iter_spans([&](std::span<uint16_t> data) {
        out.insert(out.end(), data.begin(), data.end());
    });
    ASSERT_EQ(out, std::vector<uint16_t>{3});
}

TEST(grow_buf, grab_writeable__rewind_tail) {
    MonotonicArena<uint16_t, 4> buf;
    auto *out = buf.grab_writeable(20);
    *out++ = 3;
    *out++ = 4;
    *out++ = 5;
    *out++ = 6;
    ASSERT_EQ(buf.size(), 20u);
    buf.rewind_tail(out);
    ASSERT_EQ(buf.size(), 4u);
    auto *out2 = buf.grab_writeable(3);
    ASSERT_EQ(out, out2);
    *out2++ = 7;
    *out2++ = 8;
    buf.rewind_tail(out2);
    std::vector<uint16_t> actual;
    buf.iter_spans([&](std::span<uint16_t> data) {
        actual.insert(actual.end(), data.begin(), data.end());
    });
    ASSERT_EQ(actual, (std::vector<uint16_t>{3, 4, 5, 6, 7, 8}));
}

TEST(grow_buf, push_back_many_buf) {
    MonotonicArena<uint16_t, 4> buf;
    for (size_t k = 0; k < 53; k++) {
        *buf.grab_writeable(1) = k;
    }

    MonotonicArena<uint16_t, 4> buf2;
    buf2.push_back_many(buf);
    buf2.push_back_many_reversed(buf);

    std::vector<uint16_t> out;
    buf2.iter_spans([&](std::span<uint16_t> data) {
        out.insert(out.end(), data.begin(), data.end());
    });
    ASSERT_EQ(out.size(), 106u);
    for (size_t k = 0; k < 106; k++) {
        ASSERT_EQ(out[k], k >= 53 ? 105 - k : k);
    }
}
