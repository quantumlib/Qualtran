#include "array_z.h"

#include "gtest/gtest.h"

using namespace kickmix;

TEST(qcarray, from_qubits) {
    std::vector<QubitId> qubits;
    std::vector<QubitOrBitOrBool> mixed;
    for (uint32_t k = 0; k < 100; k++) {
        qubits.push_back(QubitId{k});
        mixed.push_back(QubitId{k});
    }
    array_z array = array_z::copy_of(qubits);
    std::vector<QubitOrBitOrBool> actual;
    for (auto e : array) {
        actual.push_back(e);
    }
    ASSERT_EQ(actual, mixed);
}

TEST(qcarray, from_bits) {
    std::vector<BitId> bits;
    std::vector<QubitOrBitOrBool> mixed;
    for (uint32_t k = 0; k < 100; k++) {
        bits.push_back(BitId{k});
        mixed.push_back(BitId{k});
    }
    array_z array = array_z::copy_of(bits);
    std::vector<QubitOrBitOrBool> actual;
    for (auto e : array) {
        actual.push_back(e);
    }
    ASSERT_EQ(actual, mixed);
}

TEST(qcarray, from_mixed) {
    std::vector<QubitOrBitOrBool> mixed;
    for (uint32_t k = 0; k < 100; k++) {
        mixed.push_back(BitId{k});
    }
    for (uint32_t k = 103; k < 200; k++) {
        mixed.push_back(QubitId{k});
    }
    for (uint32_t k = 0; k < 10; k++) {
        mixed.push_back(k % 3 == 0);
    }
    array_z array = array_z::copy_of(mixed);
    std::vector<QubitOrBitOrBool> actual;
    for (auto e : array) {
        actual.push_back(e);
    }
    ASSERT_EQ(actual, mixed);
}

TEST(qcarray, skip) {
    std::vector<QubitId> qubits;
    array_z array = array_z::alloc_noinit(20, QXZTypeTag8::QUBIT_ID);
    for (uint32_t k = 0; k < 20; k++) {
        array[k] = QubitId{k};
    }
    ASSERT_EQ(
        array.as_ptr().skip(0).str(),
        "stride_span_z{q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16, q17, q18, q19}");
    ASSERT_EQ(
        array.as_ptr().skip(1).str(),
        "stride_span_z{q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16, q17, q18, q19}");
    ASSERT_EQ(array.as_ptr().skip(10).str(), "stride_span_z{q10, q11, q12, q13, q14, q15, q16, q17, q18, q19}");
    ASSERT_EQ(array.as_ptr().skip(19).str(), "stride_span_z{q19}");
    ASSERT_EQ(array.as_ptr().skip(20).str(), "stride_span_z{}");
    ASSERT_EQ(array.as_ptr().skip(21).str(), "stride_span_z{}");
}

TEST(qcarray, keep) {
    std::vector<QubitId> qubits;
    array_z array = array_z::alloc_noinit(20, QXZTypeTag8::QUBIT_ID);
    for (uint32_t k = 0; k < 20; k++) {
        array[k] = QubitId{k};
    }
    ASSERT_EQ(array.as_ptr().keep(0).str(), "stride_span_z{}");
    ASSERT_EQ(array.as_ptr().keep(1).str(), "stride_span_z{q0}");
    ASSERT_EQ(array.as_ptr().keep(10).str(), "stride_span_z{q0, q1, q2, q3, q4, q5, q6, q7, q8, q9}");
    ASSERT_EQ(
        array.as_ptr().keep(19).str(),
        "stride_span_z{q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16, q17, q18}");
    ASSERT_EQ(
        array.as_ptr().keep(20).str(),
        "stride_span_z{q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16, q17, q18, q19}");
    ASSERT_EQ(
        array.as_ptr().keep(21).str(),
        "stride_span_z{q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16, q17, q18, q19}");
}

TEST(qcarray, skip_last) {
    std::vector<QubitId> qubits;
    array_z array = array_z::alloc_noinit(20, QXZTypeTag8::QUBIT_ID);
    for (uint32_t k = 0; k < 20; k++) {
        array[k] = QubitId{k};
    }
    ASSERT_EQ(array.as_ptr().skip_last(21).str(), "stride_span_z{}");
    ASSERT_EQ(array.as_ptr().skip_last(20).str(), "stride_span_z{}");
    ASSERT_EQ(array.as_ptr().skip_last(19).str(), "stride_span_z{q0}");
    ASSERT_EQ(array.as_ptr().skip_last(10).str(), "stride_span_z{q0, q1, q2, q3, q4, q5, q6, q7, q8, q9}");
    ASSERT_EQ(
        array.as_ptr().skip_last(1).str(),
        "stride_span_z{q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16, q17, q18}");
    ASSERT_EQ(
        array.as_ptr().skip_last(0).str(),
        "stride_span_z{q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16, q17, q18, q19}");
}

TEST(qcarray, keep_last) {
    std::vector<QubitId> qubits;
    array_z array = array_z::alloc_noinit(20, QXZTypeTag8::QUBIT_ID);
    for (uint32_t k = 0; k < 20; k++) {
        array[k] = QubitId{k};
    }
    ASSERT_EQ(array.as_ptr().keep_last(0).str(), "stride_span_z{}");
    ASSERT_EQ(array.as_ptr().keep_last(1).str(), "stride_span_z{q19}");
    ASSERT_EQ(array.as_ptr().keep_last(10).str(), "stride_span_z{q10, q11, q12, q13, q14, q15, q16, q17, q18, q19}");
    ASSERT_EQ(
        array.as_ptr().keep_last(19).str(),
        "stride_span_z{q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16, q17, q18, q19}");
    ASSERT_EQ(
        array.as_ptr().keep_last(20).str(),
        "stride_span_z{q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16, q17, q18, q19}");
    ASSERT_EQ(
        array.as_ptr().keep_last(21).str(),
        "stride_span_z{q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16, q17, q18, q19}");
}

TEST(qcarray, cast) {
    array_z array = array_z::alloc_noinit(4, QXZTypeTag8::BOOL_VAL);
    array[0] = true;
    array[1] = true;
    array[2] = false;
    array[3] = true;
    stride_span<const bool> bools = (stride_span<const bool>)array.as_ptr();
    ASSERT_EQ(bools[0], true);
    ASSERT_EQ(bools[1], true);
    ASSERT_EQ(bools[2], false);
    ASSERT_EQ(bools[3], true);

    stride_span<const bool> bools2 = (stride_span<const bool>)array.as_ptr().skip(1);
    ASSERT_EQ(bools2[0], true);
    ASSERT_EQ(bools2[1], false);
    ASSERT_EQ(bools2[2], true);
}
