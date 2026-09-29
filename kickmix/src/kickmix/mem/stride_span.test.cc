#include "kickmix/mem/stride_span.h"

#include "gtest/gtest.h"

using namespace kickmix;

TEST(stride_span, iter) {
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
    stride_span<int> x(&vals[1], 2, 4);
    std::vector<int> out;
    for (const auto &e : x) {
        out.push_back(e);
    }
    ASSERT_EQ(out, (std::vector<int>{1, 3, 5, 7}));
}

TEST(stride_span, access) {
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
    stride_span<int> x(&vals[1], 2, 4);
    ASSERT_EQ(x[0], 1);
    ASSERT_EQ(x[1], 3);
    ASSERT_EQ(x[2], 5);
    ASSERT_EQ(x[3], 7);
    ASSERT_EQ(x.at(0), 1);
    ASSERT_EQ(x.at(1), 3);
    ASSERT_EQ(x.at(2), 5);
    ASSERT_EQ(x.at(3), 7);
    ASSERT_THROW({ x.at(-1); }, std::out_of_range);
    ASSERT_THROW({ x.at(4); }, std::out_of_range);
}

TEST(stride_span, collection_methods) {
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
    stride_span<int> x(&vals[1], 2, 4);
    stride_span<int> x0(&vals[1], 2, 0);
    ASSERT_EQ(x.data(), &vals[1]);
    ASSERT_EQ(x.empty(), false);
    ASSERT_EQ(x0.empty(), true);
    ASSERT_EQ(x.size(), 4);
}

TEST(stride_span, reversed) {
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
    stride_span<int> x(&vals[1], 2, 4);
    auto y = x.reversed();
    ASSERT_EQ(y.stride, -2);
    ASSERT_EQ(y.count, 4);
    ASSERT_EQ(y.ptr, &vals[7]);

    std::vector<int> out;
    for (const auto &e : y) {
        out.push_back(e);
    }
    ASSERT_EQ(out, (std::vector<int>{7, 5, 3, 1}));
}

TEST(stride_span, skip) {
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
    stride_span<int> x(&vals[1], 2, 4);
    auto y = x.skip(0);
    ASSERT_EQ(y.ptr, &vals[1]);
    ASSERT_EQ(y.count, 4);
    ASSERT_EQ(y.stride, 2);

    y = x.skip(1);
    ASSERT_EQ(y.ptr, &vals[3]);
    ASSERT_EQ(y.count, 3);
    ASSERT_EQ(y.stride, 2);

    y = x.skip(2);
    ASSERT_EQ(y.ptr, &vals[5]);
    ASSERT_EQ(y.count, 2);
    ASSERT_EQ(y.stride, 2);

    y = x.skip(3);
    ASSERT_EQ(y.ptr, &vals[7]);
    ASSERT_EQ(y.count, 1);
    ASSERT_EQ(y.stride, 2);

    y = x.skip(4);
    ASSERT_EQ(y.ptr, &vals[9]);
    ASSERT_EQ(y.count, 0);
    ASSERT_EQ(y.stride, 2);

    y = x.skip(5);
    ASSERT_EQ(y.ptr, &vals[9]);
    ASSERT_EQ(y.count, 0);
    ASSERT_EQ(y.stride, 2);

    y = x.skip(1000);
    ASSERT_EQ(y.ptr, &vals[9]);
    ASSERT_EQ(y.count, 0);
    ASSERT_EQ(y.stride, 2);
}

TEST(stride_span, keep) {
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
    stride_span<int> x(&vals[1], 2, 4);
    auto y = x.keep(0);
    ASSERT_EQ(y.ptr, &vals[1]);
    ASSERT_EQ(y.count, 0);
    ASSERT_EQ(y.stride, 2);

    y = x.keep(1);
    ASSERT_EQ(y.ptr, &vals[1]);
    ASSERT_EQ(y.count, 1);
    ASSERT_EQ(y.stride, 2);

    y = x.keep(2);
    ASSERT_EQ(y.ptr, &vals[1]);
    ASSERT_EQ(y.count, 2);
    ASSERT_EQ(y.stride, 2);

    y = x.keep(3);
    ASSERT_EQ(y.ptr, &vals[1]);
    ASSERT_EQ(y.count, 3);
    ASSERT_EQ(y.stride, 2);

    y = x.keep(4);
    ASSERT_EQ(y.ptr, &vals[1]);
    ASSERT_EQ(y.count, 4);
    ASSERT_EQ(y.stride, 2);

    y = x.keep(5);
    ASSERT_EQ(y.ptr, &vals[1]);
    ASSERT_EQ(y.count, 4);
    ASSERT_EQ(y.stride, 2);

    y = x.keep(1000);
    ASSERT_EQ(y.ptr, &vals[1]);
    ASSERT_EQ(y.count, 4);
    ASSERT_EQ(y.stride, 2);
}

TEST(stride_span, skip_last) {
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
    stride_span<int> x(&vals[1], 2, 4);
    auto y = x.skip_last(0);
    ASSERT_EQ(y.ptr, &vals[1]);
    ASSERT_EQ(y.count, 4);
    ASSERT_EQ(y.stride, 2);

    y = x.skip_last(1);
    ASSERT_EQ(y.ptr, &vals[1]);
    ASSERT_EQ(y.count, 3);
    ASSERT_EQ(y.stride, 2);

    y = x.skip_last(2);
    ASSERT_EQ(y.ptr, &vals[1]);
    ASSERT_EQ(y.count, 2);
    ASSERT_EQ(y.stride, 2);

    y = x.skip_last(3);
    ASSERT_EQ(y.ptr, &vals[1]);
    ASSERT_EQ(y.count, 1);
    ASSERT_EQ(y.stride, 2);

    y = x.skip_last(4);
    ASSERT_EQ(y.ptr, &vals[1]);
    ASSERT_EQ(y.count, 0);
    ASSERT_EQ(y.stride, 2);

    y = x.skip_last(5);
    ASSERT_EQ(y.ptr, &vals[1]);
    ASSERT_EQ(y.count, 0);
    ASSERT_EQ(y.stride, 2);

    y = x.skip_last(1000);
    ASSERT_EQ(y.ptr, &vals[1]);
    ASSERT_EQ(y.count, 0);
    ASSERT_EQ(y.stride, 2);
}

TEST(stride_span, keep_last) {
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
    stride_span<int> x(&vals[1], 2, 4);
    auto y = x.keep_last(0);
    ASSERT_EQ(y.ptr, &vals[9]);
    ASSERT_EQ(y.count, 0);
    ASSERT_EQ(y.stride, 2);

    y = x.keep_last(1);
    ASSERT_EQ(y.ptr, &vals[7]);
    ASSERT_EQ(y.count, 1);
    ASSERT_EQ(y.stride, 2);

    y = x.keep_last(2);
    ASSERT_EQ(y.ptr, &vals[5]);
    ASSERT_EQ(y.count, 2);
    ASSERT_EQ(y.stride, 2);

    y = x.keep_last(3);
    ASSERT_EQ(y.ptr, &vals[3]);
    ASSERT_EQ(y.count, 3);
    ASSERT_EQ(y.stride, 2);

    y = x.keep_last(4);
    ASSERT_EQ(y.ptr, &vals[1]);
    ASSERT_EQ(y.count, 4);
    ASSERT_EQ(y.stride, 2);

    y = x.keep_last(5);
    ASSERT_EQ(y.ptr, &vals[1]);
    ASSERT_EQ(y.count, 4);
    ASSERT_EQ(y.stride, 2);

    y = x.keep_last(1000);
    ASSERT_EQ(y.ptr, &vals[1]);
    ASSERT_EQ(y.count, 4);
    ASSERT_EQ(y.stride, 2);
}
