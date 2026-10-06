#include "mutable_circuit.h"

#include "gtest/gtest.h"

#include "kickmix/circuit/circuit.h"
#include "kickmix/util/circuit_testing.test.h"
#include "test.test.h"

using namespace kickmix;

TEST(mutable_circuit, parse_each_operation) {
    auto parse_into_vec = [](std::string_view text) {
        MutableCircuit c;
        c.append_from_kmx_text(text);
        std::vector<Op> result;
        c.to_validated_circuit().iter_ops([&](Op op) {
            result.push_back(op);
        });
        EXPECT_TRUE(c.register_data.empty()) << text;
        return result;
    };

    EXPECT_EQ(parse_into_vec(""), (std::vector<Op>{}));
    EXPECT_EQ(parse_into_vec("\n\n\n\n\n\r\n\r\n"), (std::vector<Op>{}));
    EXPECT_EQ(parse_into_vec("# test\n"), (std::vector<Op>{}));

    EXPECT_EQ(
        parse_into_vec("X q3"),
        (std::vector<Op>{Op{
            .kind = OpType::X,
            .q_control2 = {},
            .q_control1 = {},
            .q_target = QubitId{3},
            .c_target = {},
            .c_condition = {},
        }}));

    EXPECT_EQ(
        parse_into_vec("    \t  X    \t   q3   \t  # comment"),
        (std::vector<Op>{Op{
            .kind = OpType::X,
            .q_control2 = {},
            .q_control1 = {},
            .q_target = QubitId{3},
            .c_target = {},
            .c_condition = {},
        }}));

    EXPECT_EQ(
        parse_into_vec("X q3 if b5"),
        (std::vector<Op>{Op{
            .kind = OpType::X_IF,
            .q_control2 = {},
            .q_control1 = {},
            .q_target = QubitId{3},
            .c_target = {},
            .c_condition = BitId{5},
        }}));

    EXPECT_EQ(
        parse_into_vec("Z q3"),
        (std::vector<Op>{Op{
            .kind = OpType::Z,
            .q_control2 = {},
            .q_control1 = {},
            .q_target = QubitId{3},
            .c_target = {},
            .c_condition = {},
        }}));

    EXPECT_EQ(
        parse_into_vec("Z q3 if b5"),
        (std::vector<Op>{Op{
            .kind = OpType::Z_IF,
            .q_control2 = {},
            .q_control1 = {},
            .q_target = QubitId{3},
            .c_target = {},
            .c_condition = BitId{5},
        }}));

    EXPECT_EQ(
        parse_into_vec("CX q3 q7"),
        (std::vector<Op>{Op{
            .kind = OpType::CX,
            .q_control2 = {},
            .q_control1 = QubitId{3},
            .q_target = QubitId{7},
            .c_target = {},
            .c_condition = {},
        }}));

    EXPECT_EQ(
        parse_into_vec("CX q3 q7 if b5"),
        (std::vector<Op>{Op{
            .kind = OpType::CX_IF,
            .q_control2 = {},
            .q_control1 = QubitId{3},
            .q_target = QubitId{7},
            .c_target = {},
            .c_condition = BitId{5},
        }}));

    EXPECT_EQ(
        parse_into_vec("CZ q3 q7"),
        (std::vector<Op>{Op{
            .kind = OpType::CZ,
            .q_control2 = {},
            .q_control1 = QubitId{3},
            .q_target = QubitId{7},
            .c_target = {},
            .c_condition = {},
        }}));

    EXPECT_EQ(
        parse_into_vec("CZ q3 q7 if b5"),
        (std::vector<Op>{Op{
            .kind = OpType::CZ_IF,
            .q_control2 = {},
            .q_control1 = QubitId{3},
            .q_target = QubitId{7},
            .c_target = {},
            .c_condition = BitId{5},
        }}));

    EXPECT_EQ(
        parse_into_vec("CCX q3 q7 q11"),
        (std::vector<Op>{Op{
            .kind = OpType::CCX,
            .q_control2 = QubitId{3},
            .q_control1 = QubitId{7},
            .q_target = QubitId{11},
            .c_target = {},
            .c_condition = {},
        }}));

    EXPECT_EQ(
        parse_into_vec("CCX q3 q7 q11 if b5"),
        (std::vector<Op>{Op{
            .kind = OpType::CCX_IF,
            .q_control2 = QubitId{3},
            .q_control1 = QubitId{7},
            .q_target = QubitId{11},
            .c_target = {},
            .c_condition = BitId{5},
        }}));

    EXPECT_EQ(
        parse_into_vec("CCZ q3 q7 q11"),
        (std::vector<Op>{Op{
            .kind = OpType::CCZ,
            .q_control2 = QubitId{3},
            .q_control1 = QubitId{7},
            .q_target = QubitId{11},
            .c_target = {},
            .c_condition = {},
        }}));

    EXPECT_EQ(
        parse_into_vec("CCZ q3 q7 q11 if b5"),
        (std::vector<Op>{Op{
            .kind = OpType::CCZ_IF,
            .q_control2 = QubitId{3},
            .q_control1 = QubitId{7},
            .q_target = QubitId{11},
            .c_target = {},
            .c_condition = BitId{5},
        }}));

    EXPECT_EQ(
        parse_into_vec("NEG"),
        (std::vector<Op>{Op{
            .kind = OpType::NEG,
            .q_control2 = {},
            .q_control1 = {},
            .q_target = {},
            .c_target = {},
            .c_condition = {},
        }}));

    EXPECT_EQ(
        parse_into_vec("NEG if b5"),
        (std::vector<Op>{Op{
            .kind = OpType::NEG_IF,
            .q_control2 = {},
            .q_control1 = {},
            .q_target = {},
            .c_target = {},
            .c_condition = BitId{5},
        }}));

    EXPECT_EQ(
        parse_into_vec("SWAP q11 q13"),
        (std::vector<Op>{Op{
            .kind = OpType::SWAP,
            .q_control2 = {},
            .q_control1 = QubitId{11},
            .q_target = QubitId{13},
            .c_target = {},
            .c_condition = {},
        }}));

    EXPECT_EQ(
        parse_into_vec("SWAP q11 q13 if b5"),
        (std::vector<Op>{Op{
            .kind = OpType::SWAP_IF,
            .q_control2 = {},
            .q_control1 = QubitId{11},
            .q_target = QubitId{13},
            .c_target = {},
            .c_condition = BitId{5},
        }}));

    EXPECT_EQ(
        parse_into_vec("R q2"),
        (std::vector<Op>{Op{
            .kind = OpType::R,
            .q_control2 = {},
            .q_control1 = {},
            .q_target = QubitId{2},
            .c_target = {},
            .c_condition = {},
        }}));

    EXPECT_EQ(
        parse_into_vec("R q2 if b5"),
        (std::vector<Op>{Op{
            .kind = OpType::R_IF,
            .q_control2 = {},
            .q_control1 = {},
            .q_target = QubitId{2},
            .c_target = {},
            .c_condition = BitId{5},
        }}));

    EXPECT_EQ(
        parse_into_vec("HMR q2 b3"),
        (std::vector<Op>{Op{
            .kind = OpType::HMR,
            .q_control2 = {},
            .q_control1 = {},
            .q_target = QubitId{2},
            .c_target = BitId{3},
            .c_condition = {},
        }}));

    EXPECT_EQ(
        parse_into_vec("HMR q2 b3 if b5"),
        (std::vector<Op>{Op{
            .kind = OpType::HMR_IF,
            .q_control2 = {},
            .q_control1 = {},
            .q_target = QubitId{2},
            .c_target = BitId{3},
            .c_condition = BitId{5},
        }}));

    EXPECT_EQ(
        parse_into_vec("BIT_INVERT b3"),
        (std::vector<Op>{Op{
            .kind = OpType::BIT_INVERT,
            .q_control2 = {},
            .q_control1 = {},
            .q_target = {},
            .c_target = BitId{3},
            .c_condition = {},
        }}));

    EXPECT_EQ(
        parse_into_vec("BIT_INVERT b3 if b7"),
        (std::vector<Op>{Op{
            .kind = OpType::BIT_INVERT_IF,
            .q_control2 = {},
            .q_control1 = {},
            .q_target = {},
            .c_target = BitId{3},
            .c_condition = BitId{7},
        }}));

    EXPECT_EQ(
        parse_into_vec("BIT_STORE0 b3"),
        (std::vector<Op>{Op{
            .kind = OpType::BIT_STORE0,
            .q_control2 = {},
            .q_control1 = {},
            .q_target = {},
            .c_target = BitId{3},
            .c_condition = {},
        }}));

    EXPECT_EQ(
        parse_into_vec("BIT_STORE0 b3 if b7"),
        (std::vector<Op>{Op{
            .kind = OpType::BIT_STORE0_IF,
            .q_control2 = {},
            .q_control1 = {},
            .q_target = {},
            .c_target = BitId{3},
            .c_condition = BitId{7},
        }}));

    EXPECT_EQ(
        parse_into_vec("BIT_STORE1 b3"),
        (std::vector<Op>{Op{
            .kind = OpType::BIT_STORE1,
            .q_control2 = {},
            .q_control1 = {},
            .q_target = {},
            .c_target = BitId{3},
            .c_condition = {},
        }}));

    EXPECT_EQ(
        parse_into_vec("BIT_STORE1 b3 if b7"),
        (std::vector<Op>{Op{
            .kind = OpType::BIT_STORE1_IF,
            .q_control2 = {},
            .q_control1 = {},
            .q_target = {},
            .c_target = BitId{3},
            .c_condition = BitId{7},
        }}));

    EXPECT_EQ(
        parse_into_vec("PUSH_CONDITION if b3"),
        (std::vector<Op>{Op{
            .kind = OpType::PUSH_CONDITION,
            .q_control2 = {},
            .q_control1 = {},
            .q_target = {},
            .c_target = {},
            .c_condition = BitId{3},
        }}));

    EXPECT_EQ(
        parse_into_vec("POP_CONDITION"),
        (std::vector<Op>{Op{
            .kind = OpType::POP_CONDITION,
            .q_control2 = {},
            .q_control1 = {},
            .q_target = {},
            .c_target = {},
            .c_condition = {},
        }}));

    EXPECT_EQ(
        parse_into_vec("DEBUG_PRINT"),
        (std::vector<Op>{Op{
            .kind = OpType::DEBUG_PRINT_EMPTY,
            .q_control2 = {},
            .q_control1 = {},
            .q_target = {},
            .c_target = {},
            .c_condition = {},
        }}));

    EXPECT_EQ(
        parse_into_vec("DEBUG_PRINT if b5"),
        (std::vector<Op>{Op{
            .kind = OpType::DEBUG_PRINT_EMPTY_IF,
            .q_control2 = {},
            .q_control1 = {},
            .q_target = {},
            .c_target = {},
            .c_condition = BitId{5},
        }}));

    EXPECT_EQ(
        parse_into_vec("DEBUG_PRINT q2 q3 b5"),
        (std::vector<Op>{
            Op{
                .kind = OpType::DEBUG_PRINT_Q,
                .q_control2 = {},
                .q_control1 = {},
                .q_target = QubitId{2},
                .c_target = {},
                .c_condition = {},
            },
            Op{
                .kind = OpType::DEBUG_PRINT_Q,
                .q_control2 = {},
                .q_control1 = {},
                .q_target = QubitId{3},
                .c_target = {},
                .c_condition = {},
            },
            Op{
                .kind = OpType::DEBUG_PRINT_C,
                .q_control2 = {},
                .q_control1 = {},
                .q_target = {},
                .c_target = BitId{5},
                .c_condition = {},
            },
        }));

    EXPECT_EQ(
        parse_into_vec("DEBUG_PRINT q2 q3 b5 if b7"),
        (std::vector<Op>{
            Op{
                .kind = OpType::DEBUG_PRINT_Q_IF,
                .q_control2 = {},
                .q_control1 = {},
                .q_target = QubitId{2},
                .c_target = {},
                .c_condition = BitId{7},
            },
            Op{
                .kind = OpType::DEBUG_PRINT_Q_IF,
                .q_control2 = {},
                .q_control1 = {},
                .q_target = QubitId{3},
                .c_target = {},
                .c_condition = BitId{7},
            },
            Op{
                .kind = OpType::DEBUG_PRINT_C_IF,
                .q_control2 = {},
                .q_control1 = {},
                .q_target = {},
                .c_target = BitId{5},
                .c_condition = BitId{7},
            },
        }));
}

TEST(mutable_circuit, fail_parsing) {
    MutableCircuit c;
    EXPECT_THROW({ c.append_from_kmx_text("UNKNOWN"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("UNKNOWN q0"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("x q0"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("X"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("CZ q0"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("CCX q0 q1"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("CCX q0 b1 q2"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("CCX r0 b1 q2"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("CCZ b0 q1 q2"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("CCZ"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("CCZ q0 q1 q2 when test"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("PUSH_CONDITION"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("POP_CONDITION if b1"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("DEBUG_PRINT x0"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("DEBUG_PRINT xx"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("DEBUG_PRINT !"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("HMR q0"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("HMR b0"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("HMR q0 if b0"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("R b0"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("BIT_STORE0 q0"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("BIT_STORE1 q0"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("BIT_INVERT q0"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("Z"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("Z r0"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("Z b0"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("q0 X"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("NEG b0"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("CX b0"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("HMR b0"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("HMR b0 b0"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("HMR 0 b0"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("HMR q0 q1"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("HMR q0 MX"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("CX CX"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("test"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("CCX q0"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("NOP q0"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("CCX q0 q1"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("CCX q0 q1 b2"); }, std::invalid_argument);
}

TEST(mutable_circuit, boundary_values) {
    MutableCircuit c;
    EXPECT_NO_THROW({ c.append_from_kmx_text("PUSH_CONDITION if b499999999"); });
    EXPECT_THROW({ c.append_from_kmx_text("PUSH_CONDITION if b500000000"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("PUSH_CONDITION if b500000001"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("X q0 if b500000000"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("CZ q0 q1 if b500000000"); }, std::invalid_argument);
    EXPECT_NO_THROW({ c.append_from_kmx_text("X q499999999"); });
    EXPECT_THROW({ c.append_from_kmx_text("X q500000000"); }, std::invalid_argument);
    EXPECT_THROW({ c.append_from_kmx_text("X q500000001"); }, std::invalid_argument);
}
TEST(mutable_circuit, parse_later_fail_validation) {
    auto expect_fail_validation = [](std::string_view text) {
        MutableCircuit c;
        c.append_from_kmx_text(text);
        EXPECT_THROW({ c.to_validated_circuit(); }, std::invalid_argument);
    };
    expect_fail_validation("CX q0 q0");
    expect_fail_validation("CZ q0 q0");
    expect_fail_validation("SWAP q0 q0");
    expect_fail_validation("CCX q0 q0 q1");
    expect_fail_validation("CCX q1 q0 q0");
    expect_fail_validation("CCX q0 q1 q0");
    expect_fail_validation("CCZ q0 q0 q1");
    expect_fail_validation("CCZ q1 q0 q0");
    expect_fail_validation("CCZ q0 q1 q0");
}

TEST(mutable_circuit, fail_validation_size_mismatch) {
    MutableCircuit c;

    c.op_types.push_back(OpType::X);
    EXPECT_THROW({ c.to_validated_circuit(); }, std::invalid_argument);
    c.q0.push_back(1);
    EXPECT_NO_THROW({ c.to_validated_circuit(); });

    c.op_types.push_back(OpType::CX);
    EXPECT_THROW({ c.to_validated_circuit(); }, std::invalid_argument);
    c.qq0.push_back(1);
    EXPECT_THROW({ c.to_validated_circuit(); }, std::invalid_argument);
    c.qq1.push_back(2);
    EXPECT_NO_THROW({ c.to_validated_circuit(); });

    c.op_types.push_back(OpType::CCX);
    EXPECT_THROW({ c.to_validated_circuit(); }, std::invalid_argument);
    c.qqq0.push_back(1);
    EXPECT_THROW({ c.to_validated_circuit(); }, std::invalid_argument);
    c.qqq1.push_back(2);
    EXPECT_THROW({ c.to_validated_circuit(); }, std::invalid_argument);
    c.qqq2.push_back(3);
    EXPECT_NO_THROW({ c.to_validated_circuit(); });

    c.op_types.push_back(OpType::BIT_STORE0);
    EXPECT_THROW({ c.to_validated_circuit(); }, std::invalid_argument);
    c.b0.push_back(1);
    EXPECT_NO_THROW({ c.to_validated_circuit(); });

    c.op_types.push_back(OpType::BIT_STORE0_IF);
    EXPECT_THROW({ c.to_validated_circuit(); }, std::invalid_argument);
    c.b0.push_back(1);
    EXPECT_THROW({ c.to_validated_circuit(); }, std::invalid_argument);
    c.bc.push_back(1);
    EXPECT_NO_THROW({ c.to_validated_circuit(); });
}

TEST(mutable_circuit, append_from_kmx_file) {
    std::string_view contents = R"CIRCUIT(
        X q0  # test
        CX q0 q1

        # wonder
        CCZ q0 q1 q2 if b3
    )CIRCUIT";
    RaiiTempNamedFile tmp(contents);
    MutableCircuit c;
    FILE *file = fopen(tmp.path.c_str(), "r");
    c.append_from_kmx_file(file);
    fclose(file);
    ASSERT_EQ(c.to_validated_circuit(), Circuit(contents));
}
TEST(mutable_circuit, bulk_parse) {
    MutableCircuit c;
    c.append_from_kmx_text(R"CIRCUIT(
        APPEND_TO_REGISTER q0 r0
        APPEND_TO_REGISTER b0 r0
        REGISTER r1
        DEBUG_PRINT

        X q0
        X q1 if b0
        CX q0 q1
        CX q1 q2 if b1
        CCX q0 q1 q2
        CCX q1 q2 q3 if b2
        NEG
        NEG if b3

        Z q5
        Z q6 if b4
        CZ q5 q6
        CZ q6 q7 if b5
        CCZ q6 q7 q8
        CCZ q7 q8 q9 if b6
        HMR q9 b7
        HMR q10 b17 if b18
        R q11
        R q12 if b19
        BIT_INVERT b8
        BIT_INVERT b9 if b10
        BIT_STORE0 b11
        BIT_STORE0 b12 if b13
        BIT_STORE1 b14
        BIT_STORE1 b15 if b16

        SWAP q13 q14
        SWAP q15 q16 if b20

        PUSH_CONDITION if b21
        BIT_STORE1 b23
        X q20
        PUSH_CONDITION if b22
        BIT_STORE1 b24
        X q21
        POP_CONDITION
        POP_CONDITION
    )CIRCUIT");

    std::vector<OpType> ops;
    auto f = c.to_validated_circuit();
    ASSERT_EQ(f.num_ops, 35);
    ASSERT_EQ(f.register_data.size(), 2);
    ASSERT_EQ(f.register_data[0].contents, (std::vector<QubitOrBit>{QubitId{0}, BitId{0}}));
    ASSERT_EQ(f.register_data[1].contents, (std::vector<QubitOrBit>{}));
    OpType *e = f.op_types;
    ASSERT_EQ(*e++, OpType::DEBUG_PRINT_EMPTY);
    ASSERT_EQ(*e++, OpType::X);
    ASSERT_EQ(*e++, OpType::X_IF);
    ASSERT_EQ(*e++, OpType::CX);
    ASSERT_EQ(*e++, OpType::CX_IF);
    ASSERT_EQ(*e++, OpType::CCX);
    ASSERT_EQ(*e++, OpType::CCX_IF);
    ASSERT_EQ(*e++, OpType::NEG);
    ASSERT_EQ(*e++, OpType::NEG_IF);
    ASSERT_EQ(*e++, OpType::Z);
    ASSERT_EQ(*e++, OpType::Z_IF);
    ASSERT_EQ(*e++, OpType::CZ);
    ASSERT_EQ(*e++, OpType::CZ_IF);
    ASSERT_EQ(*e++, OpType::CCZ);
    ASSERT_EQ(*e++, OpType::CCZ_IF);
    ASSERT_EQ(*e++, OpType::HMR);
    ASSERT_EQ(*e++, OpType::HMR_IF);
    ASSERT_EQ(*e++, OpType::R);
    ASSERT_EQ(*e++, OpType::R_IF);
    ASSERT_EQ(*e++, OpType::BIT_INVERT);
    ASSERT_EQ(*e++, OpType::BIT_INVERT_IF);
    ASSERT_EQ(*e++, OpType::BIT_STORE0);
    ASSERT_EQ(*e++, OpType::BIT_STORE0_IF);
    ASSERT_EQ(*e++, OpType::BIT_STORE1);
    ASSERT_EQ(*e++, OpType::BIT_STORE1_IF);
    ASSERT_EQ(*e++, OpType::SWAP);
    ASSERT_EQ(*e++, OpType::SWAP_IF);
    ASSERT_EQ(*e++, OpType::PUSH_CONDITION);
    ASSERT_EQ(*e++, OpType::BIT_STORE1);
    ASSERT_EQ(*e++, OpType::X);
    ASSERT_EQ(*e++, OpType::PUSH_CONDITION);
    ASSERT_EQ(*e++, OpType::BIT_STORE1);
    ASSERT_EQ(*e++, OpType::X);
    ASSERT_EQ(*e++, OpType::POP_CONDITION);
    ASSERT_EQ(*e++, OpType::POP_CONDITION);
    ASSERT_EQ(e - f.op_types, 35);
}

TEST(mutable_circuit, append) {
    auto every = circuit_with_every_operation();

    MutableCircuit m1;
    m1.append(every);
    m1.register_data = every.register_data;
    ASSERT_EQ(m1.to_validated_circuit(), every);

    MutableCircuit m2;
    m2.append(m1);
    m2.register_data = m1.register_data;
    ASSERT_EQ(m2.to_validated_circuit(), every);

    MutableCircuit m3;
    m3.append_reversed(m2);
    m3.register_data = m2.register_data;
    ASSERT_NE(m3.to_validated_circuit(), every);
    MutableCircuit m4;
    m4.append_reversed(m3);
    m4.register_data = m3.register_data;
    ASSERT_EQ(m4.to_validated_circuit(), every);
}
