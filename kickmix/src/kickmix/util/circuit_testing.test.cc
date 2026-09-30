#include "kickmix/util/circuit_testing.test.h"

#include "gtest/gtest.h"

#include "kickmix/gen/adders/gen_iadd.h"
#include "kickmix/sim/sim.h"

using namespace kickmix;

static std::string undent(std::string_view text) {
    while (text.starts_with('\n')) {
        text = text.substr(1);
    }
    while (text.ends_with('\n') || text.ends_with(' ')) {
        text = text.substr(0, text.size() - 1);
    }

    size_t min_indent = SIZE_MAX;
    size_t cur_count = 0;
    for (char c : text) {
        if (c == '\n') {
            cur_count = 0;
        } else if (c == ' ' && cur_count != SIZE_MAX) {
            cur_count++;
        } else {
            if (c != '\r') {
                min_indent = std::min(min_indent, cur_count);
            }
            cur_count = SIZE_MAX;
        }
    }
    std::string result;
    size_t skip = min_indent;
    for (char c : text) {
        if (c == '\n') {
            skip = min_indent;
            result.push_back(c);
        } else if (c == ' ' && skip > 0) {
            skip--;
        } else {
            skip = 0;
            result.push_back(c);
        }
    }
    return result;
}

static std::vector<std::string_view> split_lines(std::string_view text) {
    std::vector<std::string_view> result;
    size_t start = 0;
    for (size_t k = 0; k < text.size(); k++) {
        if (text[k] == '\n') {
            result.push_back(text.substr(start, k - start));
            start = k + 1;
        }
    }
    result.push_back(text.substr(start));
    return result;
}

void kickmix::expect_circuit_has_text_diagram(const Circuit &circuit, std::string_view diagram) {
    auto actual_diagram = circuit.text_diagram();
    auto expected_diagram = undent(diagram);
    if (actual_diagram != expected_diagram) {
        auto actual_lines = split_lines(actual_diagram);
        auto expected_lines = split_lines(expected_diagram);
        while (actual_lines.size() < expected_lines.size()) {
            actual_lines.emplace_back();
        }
        while (expected_lines.size() < actual_lines.size()) {
            expected_lines.emplace_back();
        }
        std::string diff;
        for (size_t k = 0; k < actual_lines.size(); k++) {
            diff.append("        ");
            for (size_t k2 = 0;; k2++) {
                int c1 = k2 < actual_lines[k].size() ? actual_lines[k][k2] : -1;
                int c2 = k2 < expected_lines[k].size() ? expected_lines[k][k2] : -1;
                if (c1 == -1 && c2 == -1) {
                    break;
                }
                if (c1 != c2) {
                    diff.append("█");
                } else {
                    diff.push_back(static_cast<char>(c1));
                }
            }
            diff.push_back('\n');
        }

        std::stringstream ss;
        ss << "Actual diagram:\n";
        for (const auto &line : actual_lines) {
            ss << "        ";
            ss << line;
            ss << "\n";
        }
        ss << "\nDiff:\n";
        ss << diff;
        EXPECT_TRUE(false) << ss.str();
    }
}

Circuit kickmix::circuit_with_every_operation() {
    return Circuit(R"CIRCUIT(
        APPEND_TO_REGISTER q0 r0
        APPEND_TO_REGISTER b0 r0
        REGISTER r1 "test\r\n\B\Q\P"
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

        Z_POW q23 0.25
        Z_POW q24 0.125 if b25
    )CIRCUIT");
}

TEST(circuit_test_util, unindent) {
    ASSERT_EQ(undent(""), "");
    ASSERT_EQ(undent("a"), "a");
    ASSERT_EQ(undent(" "), "");
    ASSERT_EQ(undent("\n"), "");
    ASSERT_EQ(undent(" b "), "b");
    ASSERT_EQ(
        undent(R"TEXT(
        test
    )TEXT"),
        "test");
    ASSERT_EQ(
        undent(R"TEXT(
        test
            test2
    )TEXT"),
        "test\n    test2");
    ASSERT_EQ(
        undent(R"TEXT(
            test2
        test
    )TEXT"),
        "    test2\ntest");
    ASSERT_EQ(
        undent(undent(R"TEXT(
            test2
        test
    )TEXT")),
        "    test2\ntest");
    ASSERT_EQ(
        undent(undent(R"TEXT(
            test2 test3
        test
    )TEXT")),
        "    test2 test3\ntest");
}

TEST(circuit_with_every_operation, has_every_operation) {
    std::set<OpType> actual;
    auto c = circuit_with_every_operation();
    for (size_t k = 0; k < c.num_ops; k++) {
        actual.insert(c.op_types[k]);
    }

    std::set<OpType> expected;
    for (const auto e : OP_TYPE_TABLE) {
        expected.insert(e);
    }

    actual.insert(OpType::DEBUG_PRINT_C);
    actual.insert(OpType::DEBUG_PRINT_Q);
    actual.insert(OpType::DEBUG_PRINT_C_IF);
    actual.insert(OpType::DEBUG_PRINT_Q_IF);
    actual.insert(OpType::DEBUG_PRINT_EMPTY_IF);
    ASSERT_EQ(actual, expected);
}
