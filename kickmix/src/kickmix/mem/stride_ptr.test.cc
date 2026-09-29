#include "kickmix/mem/stride_ptr.h"

#include "gtest/gtest.h"

using namespace kickmix;

TEST(stride_ptr, access) {
    int vals[]{
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    };
    stride_ptr<int> x = {.ptr = &vals[1], .stride = 1};
    ASSERT_EQ(*x, 1);
    ASSERT_EQ(x[2], 3);

    stride_ptr<int> y{.ptr = &vals[3], .stride = 0};
    ASSERT_EQ(*y, 3);
    ASSERT_EQ(y[2], 3);
    ASSERT_EQ(y[1000], 3);
    ASSERT_EQ(y[-3], 3);
    y[5] = 4;
    ASSERT_EQ(y[0], 4);

    stride_ptr<int> z{.ptr = &vals[8], .stride = -1};
    ASSERT_EQ(*z, 8);
    ASSERT_EQ(z[1], 7);
    ASSERT_EQ(z[2], 6);

    stride_ptr<int> w{.ptr = &vals[0], .stride = 2};
    ASSERT_EQ(*w, 0);
    ASSERT_EQ(w[1], 2);
    ASSERT_EQ(w[2], 4);
    ASSERT_EQ(w[3], 6);
    w[4] = -1;
    ASSERT_EQ(vals[8], -1);
}

TEST(stride_ptr, offset) {
    int vals[]{
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    };
    stride_ptr<int> x = {.ptr = &vals[1], .stride = 2};
    ASSERT_EQ(*x, 1);
    ASSERT_EQ(*x++, 1);
    ASSERT_EQ(*x, 3);
    ASSERT_EQ(*++x, 5);
    ASSERT_EQ(*x, 5);
    ASSERT_EQ(x.stride, 2);

    ASSERT_EQ(*(x + 2), 9);
    ASSERT_EQ(*(x - 2), 1);
    ASSERT_EQ(*(2 + x), 9);

    auto y = x + 2;
    ASSERT_EQ(y.ptr, x.ptr + 4);
    ASSERT_EQ(y.stride, x.stride);
    y = 2 + x;
    ASSERT_EQ(y.ptr, x.ptr + 4);
    ASSERT_EQ(y.stride, x.stride);
    y = x - 2;
    ASSERT_EQ(y.ptr, x.ptr - 4);
    ASSERT_EQ(y.stride, x.stride);
}

TEST(stride_ptr, ioffset) {
    int vals[]{
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    };
    stride_ptr<int> x = {.ptr = &vals[1], .stride = 2};
    x += 3;
    ASSERT_EQ(x.ptr, &vals[7]);
    ASSERT_EQ(x.stride, 2);
    x -= 2;
    ASSERT_EQ(x.ptr, &vals[3]);
    ASSERT_EQ(x.stride, 2);
}
