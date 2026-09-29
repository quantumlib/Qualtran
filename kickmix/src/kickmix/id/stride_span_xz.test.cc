#include "stride_span_xz.h"

#include "gtest/gtest.h"

#include "array_z.h"

using namespace kickmix;

TEST(stride_span_xz, empty) {
    stride_span_xz q;
    ASSERT_EQ((void *)q.ptr, nullptr);
    ASSERT_EQ(q.stride, 0);
    ASSERT_EQ(q.count, 0);
    ASSERT_EQ(q.common_type, QXZTypeTag8::QUBIT_ID);
}

TEST(stride_span_xz, from_qubits) {
    std::vector<QubitId> v{QubitId(2), QubitId(3), QubitId(5), QubitId(7)};
    stride_span_xz q = stride_span<const QubitId>(v);
    ASSERT_EQ((void *)q.ptr, &v[0]);
    ASSERT_EQ(q.stride, 1);
    ASSERT_EQ(q.count, v.size());
    ASSERT_EQ(q.common_type, QXZTypeTag8::QUBIT_ID);
}

TEST(stride_span_xz, from_bits) {
    std::vector<BitId> v{BitId(2), BitId(3), BitId(5), BitId(7)};
    stride_span_xz q = stride_span<const BitId>(v);
    ASSERT_EQ((void *)q.ptr, &v[0]);
    ASSERT_EQ(q.stride, 1);
    ASSERT_EQ(q.count, v.size());
    ASSERT_EQ(q.common_type, QXZTypeTag8::BIT_ID);
}

TEST(stride_span_xz, from_xbits) {
    std::vector<XBitId> v{XBitId(2), XBitId(3), XBitId(5), XBitId(7)};
    stride_span_xz q(v);
    ASSERT_EQ((void *)q.ptr, &v[0]);
    ASSERT_EQ(q.stride, 1);
    ASSERT_EQ(q.count, v.size());
    ASSERT_EQ(q.common_type, QXZTypeTag8::XBIT_ID);
}

TEST(stride_span_xz, from_xbools) {
    std::vector<XBool> v{XBool(false), XBool(true), XBool(false), XBool(false)};
    stride_span_xz q(v);
    ASSERT_EQ((void *)q.ptr, &v[0]);
    ASSERT_EQ(q.stride, 1);
    ASSERT_EQ(q.count, v.size());
    ASSERT_EQ(q.common_type, QXZTypeTag8::XBOOL_VAL);
}

TEST(stride_span_xz, from_qubit_or_bit_or_bools) {
    std::vector<QubitOrBitOrBool> v{false, true, QubitId(1), BitId(2)};
    stride_span_xz q = stride_span<const QubitOrBitOrBool>(v);
    ASSERT_EQ((void *)q.ptr, &v[0]);
    ASSERT_EQ(q.stride, 1);
    ASSERT_EQ(q.count, v.size());
    ASSERT_EQ(q.common_type, QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_ZLIKE);
}

TEST(stride_span_xz, from_qubit_or_xbit_or_xbools) {
    std::vector<QubitOrXBitOrXBool> v{XBool(false), XBool(true), QubitId(1), XBitId(2)};
    stride_span_xz q(v);
    ASSERT_EQ((void *)q.ptr, &v[0]);
    ASSERT_EQ(q.stride, 1);
    ASSERT_EQ(q.count, v.size());
    ASSERT_EQ(q.common_type, QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_XLIKE);
}

TEST(stride_span_xz, from_qubit_or_xzbit_or_xzbools) {
    std::vector<QubitOrXZBitOrXZBool> v{XBool(false), XBool(true), QubitId(1), XBitId(2), false, BitId(3)};
    stride_span_xz q(v);
    ASSERT_EQ((void *)q.ptr, &v[0]);
    ASSERT_EQ(q.stride, 1);
    ASSERT_EQ(q.count, v.size());
    ASSERT_EQ(q.common_type, QXZTypeTag8::MAY_BE_MIXED);
}

TEST(stride_span_xz, repeat_false) {
    auto q = stride_span_xz::repeat_false(50);
    ASSERT_EQ(*q.ptr, false);
    ASSERT_EQ(q.stride, 0);
    ASSERT_EQ(q.count, 50);
    ASSERT_EQ(q.common_type, QXZTypeTag8::BOOL_VAL);
}

TEST(stride_span_xz, begin_end_iter) {
    std::vector<QubitOrXZBitOrXZBool> v{QubitId(0), QubitId(1), QubitId(2), QubitId(3), QubitId(4)};
    stride_span_xz q(v);

    ASSERT_EQ(q.begin().ptr, &v[0]);
    ASSERT_EQ(q.begin().stride, 1);
    ASSERT_EQ(q.begin().items_left, 5);

    std::vector<QubitOrXZBitOrXZBool> copy;
    for (const auto &e : q) {
        copy.push_back(e);
    }
    ASSERT_EQ(copy, v);

    q.stride *= 2;
    q.count = 3;
    ASSERT_EQ(q.begin().ptr, &v[0]);
    ASSERT_EQ(q.begin().stride, 2);
    ASSERT_EQ(q.begin().items_left, 3);
    std::vector<QubitOrXZBitOrXZBool> copy2;
    for (const auto &e : q) {
        copy2.push_back(e);
    }
    ASSERT_EQ(copy2, (std::vector<QubitOrXZBitOrXZBool>{QubitId(0), QubitId(2), QubitId(4)}));
}

TEST(stride_span_xz, front_back) {
    std::vector<QubitOrXZBitOrXZBool> v{QubitId(7), QubitId(1), QubitId(2), QubitId(3), QubitId(4)};
    stride_span_xz q(v);
    ASSERT_EQ(q.front(), QubitId(7));
    ASSERT_EQ(q.back(), QubitId(4));
    ASSERT_EQ(q.size(), 5);
    ASSERT_EQ(q.empty(), false);

    q.stride *= 2;
    q.count = 2;
    ASSERT_EQ(q.front(), QubitId(7));
    ASSERT_EQ(q.back(), QubitId(2));
    ASSERT_EQ(q.size(), 2);
    ASSERT_EQ(q.empty(), false);

    q.count = 0;
    ASSERT_EQ(q.size(), 0);
    ASSERT_EQ(q.empty(), true);
}

TEST(stride_span_xz, indexing) {
    std::vector<QubitOrXZBitOrXZBool> v{QubitId(7), QubitId(1), QubitId(2), QubitId(3), QubitId(4)};
    stride_span_xz q(v);
    ASSERT_EQ(q[0], QubitId(7));
    ASSERT_EQ(q[1], QubitId(1));
    ASSERT_EQ(q[4], QubitId(4));

    q.stride *= 2;
    q.count = 2;
    ASSERT_EQ(q[0], QubitId(7));
    ASSERT_EQ(q[1], QubitId(2));
}

static bool mixed_case(
    const std::function<bool(stride_span_xz)> &predicate, const std::vector<QubitOrXZBitOrXZBool> &v) {
    stride_span_xz q(v);
    EXPECT_EQ(q.common_type, QXZTypeTag8::MAY_BE_MIXED);
    return predicate(q);
}
static bool mixed_z_case(const std::function<bool(stride_span_xz)> &predicate, const std::vector<QubitOrBitOrBool> &v) {
    stride_span_xz q = stride_span<const QubitOrBitOrBool>(v);
    EXPECT_EQ(q.common_type, QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_ZLIKE);
    return predicate(q);
}
static bool mixed_x_case(
    const std::function<bool(stride_span_xz)> &predicate, const std::vector<QubitOrXBitOrXBool> &v) {
    stride_span_xz q(v);
    EXPECT_EQ(q.common_type, QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_XLIKE);
    return predicate(q);
}
static bool qubit_case(const std::function<bool(stride_span_xz)> &predicate, const std::vector<QubitId> &v) {
    return predicate(stride_span_xz(stride_span<const QubitId>(v)));
}
static bool bit_case(const std::function<bool(stride_span_xz)> &predicate, const std::vector<BitId> &v) {
    return predicate(stride_span_xz(stride_span<const BitId>(v)));
}
static bool xbit_case(const std::function<bool(stride_span_xz)> &predicate, const std::vector<XBitId> &v) {
    return predicate(stride_span_xz(v));
}
static bool xbool_case(const std::function<bool(stride_span_xz)> &predicate, const std::vector<XBool> &v) {
    return predicate(stride_span_xz(v));
}
static bool bool_case(const std::function<bool(stride_span_xz)> &predicate, const std::vector<bool> &v) {
    QubitOrXZBitOrXZBool *r = new QubitOrXZBitOrXZBool[v.size()];
    for (size_t k = 0; k < v.size(); k++) {
        r[k] = static_cast<bool>(v[k]);
    }
    bool b = predicate(stride_span_xz(r, 1, v.size(), QXZTypeTag8::BOOL_VAL));
    delete[] r;
    return b;
}

TEST(stride_span_xz, is_all_qubit_ids) {
    auto pred = [](const stride_span_xz &obj) {
        return obj.is_all_qubit_ids();
    };
    ASSERT_TRUE(mixed_case(pred, {}));
    ASSERT_TRUE(mixed_case(pred, {QubitId(0)}));
    ASSERT_FALSE(mixed_case(pred, {BitId(0)}));
    ASSERT_FALSE(mixed_case(pred, {XBitId(0)}));
    ASSERT_FALSE(mixed_case(pred, {true}));
    ASSERT_FALSE(mixed_case(pred, {XBool(true)}));
    ASSERT_FALSE(mixed_case(pred, {QubitId(0), BitId(0), XBitId(0), true, XBool(true)}));
    ASSERT_TRUE(mixed_x_case(pred, {}));
    ASSERT_TRUE(mixed_x_case(pred, {QubitId(0)}));
    ASSERT_FALSE(mixed_x_case(pred, {XBitId(0)}));
    ASSERT_FALSE(mixed_x_case(pred, {XBool(true)}));
    ASSERT_FALSE(mixed_x_case(pred, {QubitId(0), XBitId(0), XBool(true)}));
    ASSERT_TRUE(mixed_z_case(pred, {}));
    ASSERT_TRUE(mixed_z_case(pred, {QubitId(0)}));
    ASSERT_FALSE(mixed_z_case(pred, {BitId(0)}));
    ASSERT_FALSE(mixed_z_case(pred, {true}));
    ASSERT_FALSE(mixed_z_case(pred, {QubitId(0), BitId(0), true}));
    ASSERT_TRUE(qubit_case(pred, {}));
    ASSERT_TRUE(bit_case(pred, {}));
    ASSERT_TRUE(xbit_case(pred, {}));
    ASSERT_TRUE(bool_case(pred, {}));
    ASSERT_TRUE(xbool_case(pred, {}));
    ASSERT_TRUE(qubit_case(pred, {QubitId(0)}));
    ASSERT_FALSE(bit_case(pred, {BitId(0)}));
    ASSERT_FALSE(xbit_case(pred, {XBitId(0)}));
    ASSERT_FALSE(bool_case(pred, {true}));
    ASSERT_FALSE(xbool_case(pred, {XBool(true)}));
    ASSERT_TRUE(qubit_case(pred, {QubitId(0), QubitId(1)}));
    ASSERT_FALSE(bit_case(pred, {BitId(0), BitId(1)}));
    ASSERT_FALSE(xbit_case(pred, {XBitId(0), XBitId(1)}));
    ASSERT_FALSE(bool_case(pred, {true, false}));
    ASSERT_FALSE(xbool_case(pred, {XBool(true), XBool(false)}));
}

TEST(stride_span_xz, is_all_bit_ids) {
    auto pred = [](const stride_span_xz &obj) {
        return obj.is_all_bit_ids();
    };
    ASSERT_TRUE(mixed_case(pred, {}));
    ASSERT_FALSE(mixed_case(pred, {QubitId(0)}));
    ASSERT_TRUE(mixed_case(pred, {BitId(0)}));
    ASSERT_FALSE(mixed_case(pred, {XBitId(0)}));
    ASSERT_FALSE(mixed_case(pred, {true}));
    ASSERT_FALSE(mixed_case(pred, {XBool(true)}));
    ASSERT_FALSE(mixed_case(pred, {QubitId(0), BitId(0), XBitId(0), true, XBool(true)}));
    ASSERT_TRUE(mixed_x_case(pred, {}));
    ASSERT_FALSE(mixed_x_case(pred, {QubitId(0)}));
    ASSERT_FALSE(mixed_x_case(pred, {XBitId(0)}));
    ASSERT_FALSE(mixed_x_case(pred, {XBool(true)}));
    ASSERT_FALSE(mixed_x_case(pred, {QubitId(0), XBitId(0), XBool(true)}));
    ASSERT_TRUE(mixed_z_case(pred, {}));
    ASSERT_FALSE(mixed_z_case(pred, {QubitId(0)}));
    ASSERT_TRUE(mixed_z_case(pred, {BitId(0)}));
    ASSERT_FALSE(mixed_z_case(pred, {true}));
    ASSERT_FALSE(mixed_z_case(pred, {QubitId(0), BitId(0), true}));
    ASSERT_TRUE(qubit_case(pred, {}));
    ASSERT_TRUE(bit_case(pred, {}));
    ASSERT_TRUE(xbit_case(pred, {}));
    ASSERT_TRUE(bool_case(pred, {}));
    ASSERT_TRUE(xbool_case(pred, {}));
    ASSERT_FALSE(qubit_case(pred, {QubitId(0)}));
    ASSERT_TRUE(bit_case(pred, {BitId(0)}));
    ASSERT_FALSE(xbit_case(pred, {XBitId(0)}));
    ASSERT_FALSE(bool_case(pred, {true}));
    ASSERT_FALSE(xbool_case(pred, {XBool(true)}));
    ASSERT_FALSE(qubit_case(pred, {QubitId(0), QubitId(1)}));
    ASSERT_TRUE(bit_case(pred, {BitId(0), BitId(1)}));
    ASSERT_FALSE(xbit_case(pred, {XBitId(0), XBitId(1)}));
    ASSERT_FALSE(bool_case(pred, {true, false}));
    ASSERT_FALSE(xbool_case(pred, {XBool(true), XBool(false)}));
}

TEST(stride_span_xz, is_all_xbit_ids) {
    auto pred = [](const stride_span_xz &obj) {
        return obj.is_all_xbit_ids();
    };
    ASSERT_TRUE(mixed_case(pred, {}));
    ASSERT_FALSE(mixed_case(pred, {QubitId(0)}));
    ASSERT_FALSE(mixed_case(pred, {BitId(0)}));
    ASSERT_TRUE(mixed_case(pred, {XBitId(0)}));
    ASSERT_FALSE(mixed_case(pred, {true}));
    ASSERT_FALSE(mixed_case(pred, {XBool(true)}));
    ASSERT_FALSE(mixed_case(pred, {QubitId(0), BitId(0), XBitId(0), true, XBool(true)}));
    ASSERT_TRUE(mixed_x_case(pred, {}));
    ASSERT_FALSE(mixed_x_case(pred, {QubitId(0)}));
    ASSERT_TRUE(mixed_x_case(pred, {XBitId(0)}));
    ASSERT_FALSE(mixed_x_case(pred, {XBool(true)}));
    ASSERT_FALSE(mixed_x_case(pred, {QubitId(0), XBitId(0), XBool(true)}));
    ASSERT_TRUE(mixed_z_case(pred, {}));
    ASSERT_FALSE(mixed_z_case(pred, {QubitId(0)}));
    ASSERT_FALSE(mixed_z_case(pred, {BitId(0)}));
    ASSERT_FALSE(mixed_z_case(pred, {true}));
    ASSERT_FALSE(mixed_z_case(pred, {QubitId(0), BitId(0), true}));
    ASSERT_TRUE(qubit_case(pred, {}));
    ASSERT_TRUE(bit_case(pred, {}));
    ASSERT_TRUE(xbit_case(pred, {}));
    ASSERT_TRUE(bool_case(pred, {}));
    ASSERT_TRUE(xbool_case(pred, {}));
    ASSERT_FALSE(qubit_case(pred, {QubitId(0)}));
    ASSERT_FALSE(bit_case(pred, {BitId(0)}));
    ASSERT_TRUE(xbit_case(pred, {XBitId(0)}));
    ASSERT_FALSE(bool_case(pred, {true}));
    ASSERT_FALSE(xbool_case(pred, {XBool(true)}));
    ASSERT_FALSE(qubit_case(pred, {QubitId(0), QubitId(1)}));
    ASSERT_FALSE(bit_case(pred, {BitId(0), BitId(1)}));
    ASSERT_TRUE(xbit_case(pred, {XBitId(0), XBitId(1)}));
    ASSERT_FALSE(bool_case(pred, {true, false}));
    ASSERT_FALSE(xbool_case(pred, {XBool(true), XBool(false)}));
}

TEST(stride_span_xz, is_all_bools) {
    auto pred = [](const stride_span_xz &obj) {
        return obj.is_all_bools();
    };
    ASSERT_TRUE(mixed_case(pred, {}));
    ASSERT_FALSE(mixed_case(pred, {QubitId(0)}));
    ASSERT_FALSE(mixed_case(pred, {BitId(0)}));
    ASSERT_FALSE(mixed_case(pred, {XBitId(0)}));
    ASSERT_TRUE(mixed_case(pred, {true}));
    ASSERT_FALSE(mixed_case(pred, {XBool(true)}));
    ASSERT_FALSE(mixed_case(pred, {QubitId(0), BitId(0), XBitId(0), true, XBool(true)}));
    ASSERT_TRUE(mixed_x_case(pred, {}));
    ASSERT_FALSE(mixed_x_case(pred, {QubitId(0)}));
    ASSERT_FALSE(mixed_x_case(pred, {XBitId(0)}));
    ASSERT_FALSE(mixed_x_case(pred, {XBool(true)}));
    ASSERT_FALSE(mixed_x_case(pred, {QubitId(0), XBitId(0), XBool(true)}));
    ASSERT_TRUE(mixed_z_case(pred, {}));
    ASSERT_FALSE(mixed_z_case(pred, {QubitId(0)}));
    ASSERT_FALSE(mixed_z_case(pred, {BitId(0)}));
    ASSERT_TRUE(mixed_z_case(pred, {true}));
    ASSERT_FALSE(mixed_z_case(pred, {QubitId(0), BitId(0), true}));
    ASSERT_TRUE(qubit_case(pred, {}));
    ASSERT_TRUE(bit_case(pred, {}));
    ASSERT_TRUE(xbit_case(pred, {}));
    ASSERT_TRUE(bool_case(pred, {}));
    ASSERT_TRUE(xbool_case(pred, {}));
    ASSERT_FALSE(qubit_case(pred, {QubitId(0)}));
    ASSERT_FALSE(bit_case(pred, {BitId(0)}));
    ASSERT_FALSE(xbit_case(pred, {XBitId(0)}));
    ASSERT_TRUE(bool_case(pred, {true}));
    ASSERT_FALSE(xbool_case(pred, {XBool(true)}));
    ASSERT_FALSE(qubit_case(pred, {QubitId(0), QubitId(1)}));
    ASSERT_FALSE(bit_case(pred, {BitId(0), BitId(1)}));
    ASSERT_FALSE(xbit_case(pred, {XBitId(0), XBitId(1)}));
    ASSERT_TRUE(bool_case(pred, {true, false}));
    ASSERT_FALSE(xbool_case(pred, {XBool(true), XBool(false)}));
}

TEST(stride_span_xz, is_all_xbools) {
    auto pred = [](const stride_span_xz &obj) {
        return obj.is_all_xbools();
    };
    ASSERT_TRUE(mixed_case(pred, {}));
    ASSERT_FALSE(mixed_case(pred, {QubitId(0)}));
    ASSERT_FALSE(mixed_case(pred, {BitId(0)}));
    ASSERT_FALSE(mixed_case(pred, {XBitId(0)}));
    ASSERT_FALSE(mixed_case(pred, {true}));
    ASSERT_TRUE(mixed_case(pred, {XBool(true)}));
    ASSERT_FALSE(mixed_case(pred, {QubitId(0), BitId(0), XBitId(0), true, XBool(true)}));
    ASSERT_TRUE(mixed_x_case(pred, {}));
    ASSERT_FALSE(mixed_x_case(pred, {QubitId(0)}));
    ASSERT_FALSE(mixed_x_case(pred, {XBitId(0)}));
    ASSERT_TRUE(mixed_x_case(pred, {XBool(true)}));
    ASSERT_FALSE(mixed_x_case(pred, {QubitId(0), XBitId(0), XBool(true)}));
    ASSERT_TRUE(mixed_z_case(pred, {}));
    ASSERT_FALSE(mixed_z_case(pred, {QubitId(0)}));
    ASSERT_FALSE(mixed_z_case(pred, {BitId(0)}));
    ASSERT_FALSE(mixed_z_case(pred, {true}));
    ASSERT_FALSE(mixed_z_case(pred, {QubitId(0), BitId(0), true}));
    ASSERT_TRUE(qubit_case(pred, {}));
    ASSERT_TRUE(bit_case(pred, {}));
    ASSERT_TRUE(xbit_case(pred, {}));
    ASSERT_TRUE(bool_case(pred, {}));
    ASSERT_TRUE(xbool_case(pred, {}));
    ASSERT_FALSE(qubit_case(pred, {QubitId(0)}));
    ASSERT_FALSE(bit_case(pred, {BitId(0)}));
    ASSERT_FALSE(xbit_case(pred, {XBitId(0)}));
    ASSERT_FALSE(bool_case(pred, {true}));
    ASSERT_TRUE(xbool_case(pred, {XBool(true)}));
    ASSERT_FALSE(qubit_case(pred, {QubitId(0), QubitId(1)}));
    ASSERT_FALSE(bit_case(pred, {BitId(0), BitId(1)}));
    ASSERT_FALSE(xbit_case(pred, {XBitId(0), XBitId(1)}));
    ASSERT_FALSE(bool_case(pred, {true, false}));
    ASSERT_TRUE(xbool_case(pred, {XBool(true), XBool(false)}));
}

TEST(stride_span_xz, is_all_xlike) {
    auto pred = [](const stride_span_xz &obj) {
        return obj.is_all_xlike();
    };
    ASSERT_TRUE(mixed_case(pred, {}));
    ASSERT_TRUE(mixed_case(pred, {QubitId(0)}));
    ASSERT_FALSE(mixed_case(pred, {BitId(0)}));
    ASSERT_TRUE(mixed_case(pred, {XBitId(0)}));
    ASSERT_FALSE(mixed_case(pred, {true}));
    ASSERT_TRUE(mixed_case(pred, {XBool(true)}));
    ASSERT_TRUE(mixed_case(pred, {XBitId(0), XBool(true)}));
    ASSERT_FALSE(mixed_case(pred, {true, XBool(true)}));
    ASSERT_TRUE(mixed_case(pred, {QubitId(0), XBool(true)}));
    ASSERT_FALSE(mixed_case(pred, {BitId(0), XBool(true)}));
    ASSERT_FALSE(mixed_case(pred, {QubitId(0), BitId(0), XBitId(0), true, XBool(true)}));
    ASSERT_TRUE(mixed_x_case(pred, {}));
    ASSERT_TRUE(mixed_x_case(pred, {QubitId(0)}));
    ASSERT_TRUE(mixed_x_case(pred, {XBitId(0)}));
    ASSERT_TRUE(mixed_x_case(pred, {XBool(true)}));
    ASSERT_TRUE(mixed_x_case(pred, {QubitId(0), XBitId(0), XBool(true)}));
    ASSERT_TRUE(mixed_z_case(pred, {}));
    ASSERT_TRUE(mixed_z_case(pred, {QubitId(0)}));
    ASSERT_FALSE(mixed_z_case(pred, {BitId(0)}));
    ASSERT_FALSE(mixed_z_case(pred, {true}));
    ASSERT_FALSE(mixed_z_case(pred, {QubitId(0), BitId(0), true}));
    ASSERT_TRUE(qubit_case(pred, {}));
    ASSERT_TRUE(bit_case(pred, {}));
    ASSERT_TRUE(xbit_case(pred, {}));
    ASSERT_TRUE(bool_case(pred, {}));
    ASSERT_TRUE(xbool_case(pred, {}));
    ASSERT_TRUE(qubit_case(pred, {QubitId(0)}));
    ASSERT_FALSE(bit_case(pred, {BitId(0)}));
    ASSERT_TRUE(xbit_case(pred, {XBitId(0)}));
    ASSERT_FALSE(bool_case(pred, {true}));
    ASSERT_TRUE(xbool_case(pred, {XBool(true)}));
    ASSERT_TRUE(qubit_case(pred, {QubitId(0), QubitId(1)}));
    ASSERT_FALSE(bit_case(pred, {BitId(0), BitId(1)}));
    ASSERT_TRUE(xbit_case(pred, {XBitId(0), XBitId(1)}));
    ASSERT_FALSE(bool_case(pred, {true, false}));
    ASSERT_TRUE(xbool_case(pred, {XBool(true), XBool(false)}));
}

TEST(stride_span_xz, is_all_zlike) {
    auto pred = [](const stride_span_xz &obj) {
        return obj.is_all_zlike();
    };
    ASSERT_TRUE(mixed_case(pred, {}));
    ASSERT_TRUE(mixed_case(pred, {QubitId(0)}));
    ASSERT_TRUE(mixed_case(pred, {BitId(0)}));
    ASSERT_FALSE(mixed_case(pred, {XBitId(0)}));
    ASSERT_TRUE(mixed_case(pred, {true}));
    ASSERT_FALSE(mixed_case(pred, {XBool(true)}));
    ASSERT_FALSE(mixed_case(pred, {XBitId(0), XBool(true)}));
    ASSERT_FALSE(mixed_case(pred, {true, XBool(true)}));
    ASSERT_TRUE(mixed_case(pred, {true, BitId(1), QubitId(5)}));
    ASSERT_FALSE(mixed_case(pred, {QubitId(0), XBool(true)}));
    ASSERT_FALSE(mixed_case(pred, {BitId(0), XBool(true)}));
    ASSERT_FALSE(mixed_case(pred, {QubitId(0), BitId(0), XBitId(0), true, XBool(true)}));
    ASSERT_TRUE(mixed_x_case(pred, {}));
    ASSERT_TRUE(mixed_x_case(pred, {QubitId(0)}));
    ASSERT_FALSE(mixed_x_case(pred, {XBitId(0)}));
    ASSERT_FALSE(mixed_x_case(pred, {XBool(true)}));
    ASSERT_FALSE(mixed_x_case(pred, {QubitId(0), XBitId(0), XBool(true)}));
    ASSERT_TRUE(mixed_z_case(pred, {}));
    ASSERT_TRUE(mixed_z_case(pred, {QubitId(0)}));
    ASSERT_TRUE(mixed_z_case(pred, {BitId(0)}));
    ASSERT_TRUE(mixed_z_case(pred, {true}));
    ASSERT_TRUE(mixed_z_case(pred, {QubitId(0), BitId(0), true}));
    ASSERT_TRUE(qubit_case(pred, {}));
    ASSERT_TRUE(bit_case(pred, {}));
    ASSERT_TRUE(xbit_case(pred, {}));
    ASSERT_TRUE(bool_case(pred, {}));
    ASSERT_TRUE(xbool_case(pred, {}));
    ASSERT_TRUE(qubit_case(pred, {QubitId(0)}));
    ASSERT_TRUE(bit_case(pred, {BitId(0)}));
    ASSERT_FALSE(xbit_case(pred, {XBitId(0)}));
    ASSERT_TRUE(bool_case(pred, {true}));
    ASSERT_FALSE(xbool_case(pred, {XBool(true)}));
    ASSERT_TRUE(qubit_case(pred, {QubitId(0), QubitId(1)}));
    ASSERT_TRUE(bit_case(pred, {BitId(0), BitId(1)}));
    ASSERT_FALSE(xbit_case(pred, {XBitId(0), XBitId(1)}));
    ASSERT_TRUE(bool_case(pred, {true, false}));
    ASSERT_FALSE(xbool_case(pred, {XBool(true), XBool(false)}));
}

TEST(stride_span_xz, is_all_classical_and_xlike) {
    auto pred = [](const stride_span_xz &obj) {
        return obj.is_all_classical_and_xlike();
    };
    ASSERT_TRUE(mixed_case(pred, {}));
    ASSERT_FALSE(mixed_case(pred, {QubitId(0)}));
    ASSERT_FALSE(mixed_case(pred, {BitId(0)}));
    ASSERT_TRUE(mixed_case(pred, {XBitId(0)}));
    ASSERT_FALSE(mixed_case(pred, {true}));
    ASSERT_TRUE(mixed_case(pred, {XBool(true)}));
    ASSERT_TRUE(mixed_case(pred, {XBitId(0), XBool(true)}));
    ASSERT_FALSE(mixed_case(pred, {true, XBool(true)}));
    ASSERT_FALSE(mixed_case(pred, {QubitId(0), XBool(true)}));
    ASSERT_FALSE(mixed_case(pred, {BitId(0), XBool(true)}));
    ASSERT_FALSE(mixed_case(pred, {QubitId(0), BitId(0), XBitId(0), true, XBool(true)}));
    ASSERT_TRUE(mixed_x_case(pred, {}));
    ASSERT_FALSE(mixed_x_case(pred, {QubitId(0)}));
    ASSERT_TRUE(mixed_x_case(pred, {XBitId(0)}));
    ASSERT_TRUE(mixed_x_case(pred, {XBool(true)}));
    ASSERT_FALSE(mixed_x_case(pred, {QubitId(0), XBitId(0), XBool(true)}));
    ASSERT_TRUE(mixed_z_case(pred, {}));
    ASSERT_FALSE(mixed_z_case(pred, {QubitId(0)}));
    ASSERT_FALSE(mixed_z_case(pred, {BitId(0)}));
    ASSERT_FALSE(mixed_z_case(pred, {true}));
    ASSERT_FALSE(mixed_z_case(pred, {QubitId(0), BitId(0), true}));
    ASSERT_TRUE(qubit_case(pred, {}));
    ASSERT_TRUE(bit_case(pred, {}));
    ASSERT_TRUE(xbit_case(pred, {}));
    ASSERT_TRUE(bool_case(pred, {}));
    ASSERT_TRUE(xbool_case(pred, {}));
    ASSERT_FALSE(qubit_case(pred, {QubitId(0)}));
    ASSERT_FALSE(bit_case(pred, {BitId(0)}));
    ASSERT_TRUE(xbit_case(pred, {XBitId(0)}));
    ASSERT_FALSE(bool_case(pred, {true}));
    ASSERT_TRUE(xbool_case(pred, {XBool(true)}));
    ASSERT_FALSE(qubit_case(pred, {QubitId(0), QubitId(1)}));
    ASSERT_FALSE(bit_case(pred, {BitId(0), BitId(1)}));
    ASSERT_TRUE(xbit_case(pred, {XBitId(0), XBitId(1)}));
    ASSERT_FALSE(bool_case(pred, {true, false}));
    ASSERT_TRUE(xbool_case(pred, {XBool(true), XBool(false)}));
}

TEST(stride_span_xz, is_all_classical_and_zlike) {
    auto pred = [](const stride_span_xz &obj) {
        return obj.is_all_classical_and_zlike();
    };
    ASSERT_TRUE(mixed_case(pred, {}));
    ASSERT_FALSE(mixed_case(pred, {QubitId(0)}));
    ASSERT_TRUE(mixed_case(pred, {BitId(0)}));
    ASSERT_FALSE(mixed_case(pred, {XBitId(0)}));
    ASSERT_TRUE(mixed_case(pred, {true}));
    ASSERT_FALSE(mixed_case(pred, {XBool(true)}));
    ASSERT_FALSE(mixed_case(pred, {XBitId(0), XBool(true)}));
    ASSERT_FALSE(mixed_case(pred, {true, XBool(true)}));
    ASSERT_FALSE(mixed_case(pred, {true, BitId(1), QubitId(5)}));
    ASSERT_FALSE(mixed_case(pred, {QubitId(0), XBool(true)}));
    ASSERT_FALSE(mixed_case(pred, {BitId(0), XBool(true)}));
    ASSERT_FALSE(mixed_case(pred, {QubitId(0), BitId(0), XBitId(0), true, XBool(true)}));
    ASSERT_TRUE(mixed_x_case(pred, {}));
    ASSERT_FALSE(mixed_x_case(pred, {QubitId(0)}));
    ASSERT_FALSE(mixed_x_case(pred, {XBitId(0)}));
    ASSERT_FALSE(mixed_x_case(pred, {XBool(true)}));
    ASSERT_FALSE(mixed_x_case(pred, {QubitId(0), XBitId(0), XBool(true)}));
    ASSERT_TRUE(mixed_z_case(pred, {}));
    ASSERT_FALSE(mixed_z_case(pred, {QubitId(0)}));
    ASSERT_TRUE(mixed_z_case(pred, {BitId(0)}));
    ASSERT_TRUE(mixed_z_case(pred, {true}));
    ASSERT_FALSE(mixed_z_case(pred, {QubitId(0), BitId(0), true}));
    ASSERT_TRUE(qubit_case(pred, {}));
    ASSERT_TRUE(bit_case(pred, {}));
    ASSERT_TRUE(xbit_case(pred, {}));
    ASSERT_TRUE(bool_case(pred, {}));
    ASSERT_TRUE(xbool_case(pred, {}));
    ASSERT_FALSE(qubit_case(pred, {QubitId(0)}));
    ASSERT_TRUE(bit_case(pred, {BitId(0)}));
    ASSERT_FALSE(xbit_case(pred, {XBitId(0)}));
    ASSERT_TRUE(bool_case(pred, {true}));
    ASSERT_FALSE(xbool_case(pred, {XBool(true)}));
    ASSERT_FALSE(qubit_case(pred, {QubitId(0), QubitId(1)}));
    ASSERT_TRUE(bit_case(pred, {BitId(0), BitId(1)}));
    ASSERT_FALSE(xbit_case(pred, {XBitId(0), XBitId(1)}));
    ASSERT_TRUE(bool_case(pred, {true, false}));
    ASSERT_FALSE(xbool_case(pred, {XBool(true), XBool(false)}));
}

TEST(stride_span_xz, checked_cast_to_qubit_ids) {
    {
        std::vector<QubitId> v{QubitId(0), QubitId(1), QubitId(2), QubitId(3), QubitId(4), QubitId(5)};
        stride_span_xz q = stride_span<const QubitId>(v);
        q.stride = 2;
        q.count = 3;
        auto s = q.checked_cast_to_qubit_ids("test");
        ASSERT_EQ(s.data(), v.data());
        ASSERT_EQ(s.stride, 2);
        ASSERT_EQ(s.size(), 3);
    }

    {
        std::vector<QubitOrXZBitOrXZBool> v{QubitId(0), QubitId(1), QubitId(2), QubitId(3), QubitId(4), QubitId(5)};
        stride_span_xz q(v);
        auto s = q.checked_cast_to_qubit_ids("test");
        ASSERT_EQ((void *)s.data(), v.data());
        ASSERT_EQ(s.stride, 1);
        ASSERT_EQ(s.size(), 6);
    }

    {
        std::vector<BitId> v{};
        stride_span_xz q = stride_span<const BitId>(v);
        auto s = q.checked_cast_to_qubit_ids("test");
        ASSERT_EQ((void *)s.data(), v.data());
        ASSERT_EQ(s.stride, 1);
        ASSERT_EQ(s.size(), 0);
    }

    EXPECT_THROW(
        { stride_span_xz(stride_span<const BitId>(std::vector<BitId>{BitId(0)})).checked_cast_to_qubit_ids("test"); },
        std::invalid_argument);
    EXPECT_THROW(
        { stride_span_xz(std::vector<QubitOrXZBitOrXZBool>{BitId(0)}).checked_cast_to_qubit_ids("test"); },
        std::invalid_argument);
}

TEST(stride_span_xz, checked_cast_to_bit_ids) {
    {
        std::vector<BitId> v{BitId(0), BitId(1), BitId(2), BitId(3), BitId(4), BitId(5)};
        stride_span_xz q = stride_span<const BitId>(v);
        q.stride = 2;
        q.count = 3;
        auto s = q.checked_cast_to_bit_ids("test");
        ASSERT_EQ(s.data(), v.data());
        ASSERT_EQ(s.stride, 2);
        ASSERT_EQ(s.size(), 3);
    }

    {
        std::vector<QubitOrXZBitOrXZBool> v{BitId(0), BitId(1), BitId(2), BitId(3), BitId(4), BitId(5)};
        stride_span_xz q(v);
        auto s = q.checked_cast_to_bit_ids("test");
        ASSERT_EQ((void *)s.data(), v.data());
        ASSERT_EQ(s.stride, 1);
        ASSERT_EQ(s.size(), 6);
    }

    {
        std::vector<QubitId> v{};
        stride_span_xz q = stride_span<const QubitId>(v);
        auto s = q.checked_cast_to_bit_ids("test");
        ASSERT_EQ((void *)s.data(), v.data());
        ASSERT_EQ(s.stride, 1);
        ASSERT_EQ(s.size(), 0);
    }

    EXPECT_THROW(
        {
            stride_span_xz(stride_span<const QubitId>(std::vector<QubitId>{QubitId(0)}))
                .checked_cast_to_bit_ids("test");
        },
        std::invalid_argument);
    EXPECT_THROW(
        { stride_span_xz(std::vector<QubitOrXZBitOrXZBool>{QubitId(0)}).checked_cast_to_bit_ids("test"); },
        std::invalid_argument);
}

TEST(stride_span_xz, checked_cast_to_xbit_ids) {
    {
        std::vector<XBitId> v{XBitId(0), XBitId(1), XBitId(2), XBitId(3), XBitId(4), XBitId(5)};
        stride_span_xz q(v);
        q.stride = 2;
        q.count = 3;
        auto s = q.checked_cast_to_xbit_ids("test");
        ASSERT_EQ(s.data(), v.data());
        ASSERT_EQ(s.stride, 2);
        ASSERT_EQ(s.size(), 3);
    }

    {
        std::vector<QubitOrXZBitOrXZBool> v{XBitId(0), XBitId(1), XBitId(2), XBitId(3), XBitId(4), XBitId(5)};
        stride_span_xz q(v);
        auto s = q.checked_cast_to_xbit_ids("test");
        ASSERT_EQ((void *)s.data(), v.data());
        ASSERT_EQ(s.stride, 1);
        ASSERT_EQ(s.size(), 6);
    }

    {
        std::vector<QubitId> v{};
        stride_span_xz q = stride_span<const QubitId>(v);
        auto s = q.checked_cast_to_xbit_ids("test");
        ASSERT_EQ((void *)s.data(), v.data());
        ASSERT_EQ(s.stride, 1);
        ASSERT_EQ(s.size(), 0);
    }

    EXPECT_THROW(
        {
            stride_span_xz(stride_span<const QubitId>(std::vector<QubitId>{QubitId(0)}))
                .checked_cast_to_xbit_ids("test");
        },
        std::invalid_argument);
    EXPECT_THROW(
        { stride_span_xz(std::vector<QubitOrXZBitOrXZBool>{QubitId(0)}).checked_cast_to_xbit_ids("test"); },
        std::invalid_argument);
}

TEST(stride_span_xz, checked_cast_to_qubit_or_bit_or_bool) {
    {
        std::vector<QubitId> v{QubitId(0), QubitId(1)};
        stride_span_xz q = stride_span<const QubitId>(v);
        q.stride = 2;
        q.count = 3;
        auto s = q.checked_cast_to_qubit_or_bit_or_bool("test");
        ASSERT_EQ((void *)s.data(), v.data());
        ASSERT_EQ(s.stride, 2);
        ASSERT_EQ(s.size(), 3);
    }

    {
        std::vector<QubitOrXZBitOrXZBool> v{QubitId(2), BitId(1), true};
        stride_span_xz q(v);
        auto s = q.checked_cast_to_qubit_or_bit_or_bool("test");
        ASSERT_EQ((void *)s.data(), v.data());
        ASSERT_EQ(s.stride, 1);
        ASSERT_EQ(s.size(), 3);
    }

    {
        std::vector<XBitId> v{};
        stride_span_xz q(v);
        auto s = q.checked_cast_to_qubit_or_bit_or_bool("test");
        ASSERT_EQ((void *)s.data(), v.data());
        ASSERT_EQ(s.stride, 1);
        ASSERT_EQ(s.size(), 0);
    }

    EXPECT_THROW(
        { stride_span_xz(std::vector<XBitId>{XBitId(0)}).checked_cast_to_qubit_or_bit_or_bool("test"); },
        std::invalid_argument);
    EXPECT_THROW(
        {
            stride_span_xz(std::vector<QubitOrXZBitOrXZBool>{XBool(false)})
                .checked_cast_to_qubit_or_bit_or_bool("test");
        },
        std::invalid_argument);
}

TEST(stride_span_xz, checked_cast_to_qubit_or_xbit_or_xbool) {
    {
        std::vector<QubitId> v{QubitId(0), QubitId(1)};
        stride_span_xz q = stride_span<const QubitId>(v);
        q.stride = 2;
        q.count = 3;
        auto s = q.checked_cast_to_qubit_or_xbit_or_xbool("test");
        ASSERT_EQ((void *)s.data(), v.data());
        ASSERT_EQ(s.stride, 2);
        ASSERT_EQ(s.size(), 3);
    }

    {
        std::vector<QubitOrXZBitOrXZBool> v{QubitId(2), XBitId(1), XBool(true)};
        stride_span_xz q(v);
        auto s = q.checked_cast_to_qubit_or_xbit_or_xbool("test");
        ASSERT_EQ((void *)s.data(), v.data());
        ASSERT_EQ(s.stride, 1);
        ASSERT_EQ(s.size(), 3);
    }

    {
        std::vector<BitId> v{};
        stride_span_xz q = stride_span<const BitId>(v);
        auto s = q.checked_cast_to_qubit_or_xbit_or_xbool("test");
        ASSERT_EQ((void *)s.data(), v.data());
        ASSERT_EQ(s.stride, 1);
        ASSERT_EQ(s.size(), 0);
    }

    EXPECT_THROW(
        {
            stride_span_xz(stride_span<const BitId>(std::vector<BitId>{BitId(0)}))
                .checked_cast_to_qubit_or_xbit_or_xbool("test");
        },
        std::invalid_argument);
    EXPECT_THROW(
        { stride_span_xz(std::vector<QubitOrXZBitOrXZBool>{false}).checked_cast_to_qubit_or_xbit_or_xbool("test"); },
        std::invalid_argument);
}

TEST(stride_span_xz, str) {
    ASSERT_EQ(
        stride_span_xz(std::vector<QubitOrXZBitOrXZBool>{QubitId(2), BitId(1), true, XBitId(3), XBool(false)}).str(),
        "stride_span_xz{q2, b1, true, xb3, XBool(false)}");
}

TEST(stride_span_xz, reversed) {
    std::vector<QubitId> v{QubitId(2), QubitId(3), QubitId(5), QubitId(7), QubitId(11), QubitId(13)};
    stride_span_xz q = stride_span<const QubitId>(v);
    stride_span_xz r = q.reversed();
    ASSERT_EQ((void *)r.ptr, &v[5]);
    ASSERT_EQ(r.stride, -1);
    ASSERT_EQ(r.count, v.size());
    ASSERT_EQ(q.common_type, QXZTypeTag8::QUBIT_ID);

    q.stride = 2;
    q.count = 3;
    r = q.reversed();
    ASSERT_EQ((void *)r.ptr, &v[4]);
    ASSERT_EQ(r.stride, -2);
    ASSERT_EQ(r.count, 3);
    ASSERT_EQ(q.common_type, QXZTypeTag8::QUBIT_ID);
}

TEST(stride_span_xz, skip) {
    std::vector<QubitId> v{QubitId(2), QubitId(3), QubitId(5), QubitId(7), QubitId(11), QubitId(13)};
    stride_span_xz q = stride_span<const QubitId>(v);
    q.stride = 2;
    q.count = 3;
    stride_span_xz r;

    r = q.skip(0);
    ASSERT_EQ(r.ptr, q.ptr);
    ASSERT_EQ(r.count, 3);
    ASSERT_EQ(r.stride, 2);

    r = q.skip(1);
    ASSERT_EQ(r.ptr, q.ptr + 2);
    ASSERT_EQ(r.count, 2);
    ASSERT_EQ(r.stride, 2);

    r = q.skip(2);
    ASSERT_EQ(r.ptr, q.ptr + 4);
    ASSERT_EQ(r.count, 1);
    ASSERT_EQ(r.stride, 2);

    r = q.skip(3);
    ASSERT_EQ(r.ptr, q.ptr + 6);
    ASSERT_EQ(r.count, 0);
    ASSERT_EQ(r.stride, 2);

    r = q.skip(4);
    ASSERT_EQ(r.ptr, q.ptr + 6);
    ASSERT_EQ(r.count, 0);
    ASSERT_EQ(r.stride, 2);
}

TEST(stride_span_xz, keep_last) {
    std::vector<QubitId> v{QubitId(2), QubitId(3), QubitId(5), QubitId(7), QubitId(11), QubitId(13)};
    stride_span_xz q = stride_span<const QubitId>(v);
    q.stride = 2;
    q.count = 3;
    stride_span_xz r;

    r = q.keep_last(0);
    ASSERT_EQ(r.ptr, q.ptr + 6);
    ASSERT_EQ(r.count, 0);
    ASSERT_EQ(r.stride, 2);

    r = q.keep_last(1);
    ASSERT_EQ(r.ptr, q.ptr + 4);
    ASSERT_EQ(r.count, 1);
    ASSERT_EQ(r.stride, 2);

    r = q.keep_last(2);
    ASSERT_EQ(r.ptr, q.ptr + 2);
    ASSERT_EQ(r.count, 2);
    ASSERT_EQ(r.stride, 2);

    r = q.keep_last(3);
    ASSERT_EQ(r.ptr, q.ptr);
    ASSERT_EQ(r.count, 3);
    ASSERT_EQ(r.stride, 2);

    r = q.keep_last(4);
    ASSERT_EQ(r.ptr, q.ptr);
    ASSERT_EQ(r.count, 3);
    ASSERT_EQ(r.stride, 2);
}

TEST(stride_span_xz, keep) {
    std::vector<QubitId> v{QubitId(2), QubitId(3), QubitId(5), QubitId(7), QubitId(11), QubitId(13)};
    stride_span_xz q = stride_span<const QubitId>(v);
    q.stride = 2;
    q.count = 3;
    stride_span_xz r;

    r = q.keep(0);
    ASSERT_EQ(r.ptr, q.ptr);
    ASSERT_EQ(r.count, 0);
    ASSERT_EQ(r.stride, 2);

    r = q.keep(1);
    ASSERT_EQ(r.ptr, q.ptr);
    ASSERT_EQ(r.count, 1);
    ASSERT_EQ(r.stride, 2);

    r = q.keep(2);
    ASSERT_EQ(r.ptr, q.ptr);
    ASSERT_EQ(r.count, 2);
    ASSERT_EQ(r.stride, 2);

    r = q.keep(3);
    ASSERT_EQ(r.ptr, q.ptr);
    ASSERT_EQ(r.count, 3);
    ASSERT_EQ(r.stride, 2);

    r = q.keep(4);
    ASSERT_EQ(r.ptr, q.ptr);
    ASSERT_EQ(r.count, 3);
    ASSERT_EQ(r.stride, 2);
}

TEST(stride_span_xz, skip_last) {
    std::vector<QubitId> v{QubitId(2), QubitId(3), QubitId(5), QubitId(7), QubitId(11), QubitId(13)};
    stride_span_xz q = stride_span<const QubitId>(v);
    q.stride = 2;
    q.count = 3;
    stride_span_xz r;

    r = q.skip_last(0);
    ASSERT_EQ(r.ptr, q.ptr);
    ASSERT_EQ(r.count, 3);
    ASSERT_EQ(r.stride, 2);

    r = q.skip_last(1);
    ASSERT_EQ(r.ptr, q.ptr);
    ASSERT_EQ(r.count, 2);
    ASSERT_EQ(r.stride, 2);

    r = q.skip_last(2);
    ASSERT_EQ(r.ptr, q.ptr);
    ASSERT_EQ(r.count, 1);
    ASSERT_EQ(r.stride, 2);

    r = q.skip_last(3);
    ASSERT_EQ(r.ptr, q.ptr);
    ASSERT_EQ(r.count, 0);
    ASSERT_EQ(r.stride, 2);

    r = q.skip_last(4);
    ASSERT_EQ(r.ptr, q.ptr);
    ASSERT_EQ(r.count, 0);
    ASSERT_EQ(r.stride, 2);
}

TEST(stride_span_xz, conv_stride_span) {
    std::vector<QubitId> v{QubitId(2), QubitId(3), QubitId(5), QubitId(7), QubitId(11), QubitId(13)};
    stride_span_xz q = stride_span<const QubitId>(v);
    q.stride = 2;
    q.count = 3;
    stride_span<const QubitOrXZBitOrXZBool> r = q;
    ASSERT_EQ((void *)r.ptr, q.ptr);
    ASSERT_EQ(r.stride, q.stride);
    ASSERT_EQ(r.count, q.count);
}

TEST(stride_span_xz, conv_stride_ptr) {
    std::vector<QubitId> v{QubitId(2), QubitId(3), QubitId(5), QubitId(7), QubitId(11), QubitId(13)};
    stride_span_xz q = stride_span<const QubitId>(v);
    q.stride = 2;
    q.count = 3;
    stride_ptr<const QubitOrXZBitOrXZBool> r = q;
    ASSERT_EQ((void *)r.ptr, q.ptr);
    ASSERT_EQ(r.stride, q.stride);
}

TEST(stride_span_xz, conv_data) {
    std::vector<QubitId> v{QubitId(2), QubitId(3), QubitId(5), QubitId(7), QubitId(11), QubitId(13)};
    stride_span_xz q = stride_span<const QubitId>(v);
    q.stride = 2;
    q.count = 3;
    stride_span<const QubitId> r = q.cast_data<const QubitId>();
    ASSERT_EQ((void *)r.ptr, q.ptr);
    ASSERT_EQ(r.stride, q.stride);
    ASSERT_EQ(r.count, q.count);
    stride_span<const uint32_t> r2 = q.cast_data<const uint32_t>();
    ASSERT_EQ((void *)r2.ptr, q.ptr);
    ASSERT_EQ(r2.stride, q.stride);
    ASSERT_EQ(r2.count, q.count);
}

TEST(stride_span_xz, compute_common_type_data) {
    ASSERT_EQ(
        stride_span_xz(std::vector<QubitOrXZBitOrXZBool>{QubitId(2), QubitId(3), QubitId(5)}).compute_common_type(),
        QXZTypeTag8::QUBIT_ID);
    ASSERT_EQ(
        stride_span_xz(std::vector<QubitOrXZBitOrXZBool>{BitId(2), BitId(3), BitId(5)}).compute_common_type(),
        QXZTypeTag8::BIT_ID);
    ASSERT_EQ(
        stride_span_xz(std::vector<QubitOrXZBitOrXZBool>{true, false}).compute_common_type(), QXZTypeTag8::BOOL_VAL);
    ASSERT_EQ(
        stride_span_xz(std::vector<QubitOrXZBitOrXZBool>{XBool(true), XBool(false)}).compute_common_type(),
        QXZTypeTag8::XBOOL_VAL);
    ASSERT_EQ(
        stride_span_xz(std::vector<QubitOrXZBitOrXZBool>{XBitId(2), XBitId(3), XBitId(5)}).compute_common_type(),
        QXZTypeTag8::XBIT_ID);

    ASSERT_EQ(
        stride_span_xz(std::vector<QubitOrXZBitOrXZBool>{QubitId(2), BitId(3), true}).compute_common_type(),
        QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_ZLIKE);
    ASSERT_EQ(
        stride_span_xz(std::vector<QubitOrXZBitOrXZBool>{QubitId(2), BitId(3)}).compute_common_type(),
        QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_ZLIKE);
    ASSERT_EQ(
        stride_span_xz(std::vector<QubitOrXZBitOrXZBool>{QubitId(2), true}).compute_common_type(),
        QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_ZLIKE);

    ASSERT_EQ(
        stride_span_xz(std::vector<QubitOrXZBitOrXZBool>{QubitId(2), XBitId(3), XBool(true)}).compute_common_type(),
        QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_XLIKE);
    ASSERT_EQ(
        stride_span_xz(std::vector<QubitOrXZBitOrXZBool>{QubitId(2), XBitId(3)}).compute_common_type(),
        QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_XLIKE);
    ASSERT_EQ(
        stride_span_xz(std::vector<QubitOrXZBitOrXZBool>{QubitId(2), XBool(true)}).compute_common_type(),
        QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_XLIKE);

    ASSERT_EQ(
        stride_span_xz(std::vector<QubitOrXZBitOrXZBool>{QubitId(2), BitId(3), XBitId(0)}).compute_common_type(),
        QXZTypeTag8::MAY_BE_MIXED);
    ASSERT_EQ(
        stride_span_xz(std::vector<QubitOrXZBitOrXZBool>{QubitId(2), BitId(3), XBitId(0), false, XBool(false)})
            .compute_common_type(),
        QXZTypeTag8::MAY_BE_MIXED);
}

TEST(stride_span_xz, has_same_contents_as) {
    std::vector<QubitOrXZBitOrXZBool> v{QubitId(2), QubitId(3), QubitId(5)};
    std::vector<QubitId> v2{QubitId(2), QubitId(3), QubitId(5), QubitId(7)};
    std::vector<QubitId> v3{QubitId(2), QubitId(5)};
    stride_span_xz q(v);
    stride_span_xz q2 = stride_span<const QubitId>(v2);
    stride_span_xz q3 = stride_span<const QubitId>(v3);
    ASSERT_FALSE(q.has_same_contents_as(q2));
    ASSERT_TRUE(q.has_same_contents_as(q2.skip_last(1)));
    ASSERT_FALSE(q.has_same_contents_as(q3));
    q.stride = 2;
    q.count = 2;
    ASSERT_TRUE(q.has_same_contents_as(q3));
    ASSERT_FALSE(q.has_same_contents_as(q2));
}

TEST(stride_span_z, preserves_stride_from_bit_and_mixed_spans) {
    std::vector<BitId> bits{BitId(1), BitId(2), BitId(3)};
    stride_span<const BitId> rev_bits = stride_span<const BitId>(bits).reversed();
    stride_span_z z_bits(rev_bits);
    ASSERT_EQ(z_bits.stride, -1);
    ASSERT_EQ(z_bits[0], QubitOrBitOrBool(BitId(3)));
    ASSERT_EQ(z_bits[2], QubitOrBitOrBool(BitId(1)));

    std::vector<QubitOrBitOrBool> mixed{QubitId(4), BitId(5), true};
    stride_span<const QubitOrBitOrBool> rev_mixed = stride_span<const QubitOrBitOrBool>(mixed).reversed();
    stride_span_z z_mixed(rev_mixed);
    ASSERT_EQ(z_mixed.stride, -1);
    ASSERT_EQ(z_mixed[0], QubitOrBitOrBool(true));
    ASSERT_EQ(z_mixed[2], QubitOrBitOrBool(QubitId(4)));
}

TEST(stride_span_z, vector_constructors_preserve_pointer) {
    std::vector<QubitId> qubits{QubitId(1), QubitId(2)};
    stride_span_z z_q(qubits);
    ASSERT_EQ(z_q.data(), reinterpret_cast<const QubitOrBitOrBool *>(qubits.data()));

    std::vector<BitId> bits{BitId(3), BitId(4)};
    stride_span_z z_b(bits);
    ASSERT_EQ(z_b.data(), reinterpret_cast<const QubitOrBitOrBool *>(bits.data()));

    std::vector<QubitOrBitOrBool> mixed{QubitId(1), BitId(2), true};
    stride_span_z z_m(mixed);
    ASSERT_EQ(z_m.data(), mixed.data());
}
