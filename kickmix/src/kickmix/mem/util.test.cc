#include "util.h"

#include "gtest/gtest.h"

#include "kickmix/mem/util.h"

using namespace kickmix;

TEST(mem_util, max_u32) {
    std::vector<uint32_t> items;
    for (size_t k = 0; k < 100; k++) {
        items.push_back(k * k % 255);
    }
    ASSERT_EQ(max_u32(items.data(), 100), 246);
    ASSERT_EQ(max_u32(items.data(), 10), 81);
    ASSERT_EQ(max_u32(items.data() + 20, 10), 229);
    ASSERT_EQ(max_u32(items.data(), 0), 0);
    ASSERT_EQ(max_u32(items.data(), 1), 0);
    ASSERT_EQ(max_u32(items.data() + 1, 1), 1);
    ASSERT_EQ(max_u32(items.data() + 2, 1), 4);
}

TEST(mem_util, memcpy_repeat) {
    std::vector<uint32_t> items;
    for (size_t k = 0; k < 100; k++) {
        items.push_back(k);
    }
    std::vector<uint32_t> dst;
    dst.resize(5);
    memcpy_repeat(dst.data(), items.data() + 3, 1 * sizeof(uint32_t), 5);
    ASSERT_EQ(dst, (std::vector<uint32_t>{3, 3, 3, 3, 3}));

    dst.resize(10);
    memcpy_repeat(dst.data(), items.data() + 3, 2 * sizeof(uint32_t), 5);
    ASSERT_EQ(dst, (std::vector<uint32_t>{3, 4, 3, 4, 3, 4, 3, 4, 3, 4}));

    dst.resize(items.size() * 64);
    memcpy_repeat(dst.data(), items.data(), items.size() * sizeof(uint32_t), 64);
    for (size_t k = 0; k < dst.size(); k++) {
        ASSERT_EQ(dst[k], items[k % items.size()]);
    }
}

TEST(mem_util, aligned_alloc_else_throw) {
    ASSERT_THROW({ (aligned_alloc_else_throw<uint32_t, 8>(SIZE_MAX)); }, std::runtime_error);

    uint32_t *v = aligned_alloc_else_throw<uint32_t, 32>(5);
    ASSERT_EQ((uintptr_t)v & 31, 0);
    v[4] = 4;
    v[0] = 0;
    // Allocation gets rounded up to a multiple of 16, so that it's safe to use simd values.
    // This implies v[7] must exist because v[8] would be at the 32-byte aligned boundary.
    v[7] = 7;

    ASSERT_EQ(v[4], 4);
    ASSERT_EQ(v[0], 0);
    ASSERT_EQ(v[7], 7);
    free_aligned(v);
}
