#include "gtest/gtest.h"

#include "kickmix/gen/comparators/gen_cmp.h"
#include "kickmix/sim/fuzzer.h"
#include "kickmix/util/circuit_testing.test.h"
#include "test.test.h"

using namespace kickmix;

TEST(circuit, text_diagram) {
    expect_circuit_has_text_diagram(
        Circuit(R"CIRCUIT(
            CCX q0 q1 q2
            CX q0 q1
            X q0
            NEG
            NEG if b1
            HMR q2 b1 if b3
        )CIRCUIT"),
        R"DIAGRAM(
                   neg
        q0: -@-@-X------------------
             | |   neg if b1
        q1: -@-X--------------------
             |               if(b3)
        q2: -X---------------HMR=b1
    )DIAGRAM");
}

TEST(circuit, text_diagram_merge_cx) {
    expect_circuit_has_text_diagram(
        Circuit(R"CIRCUIT(
            CX q0 q1
            CX q0 q2
            CX q0 q3
            CX q0 q4
            CX q0 q2
            CX q0 q4
        )CIRCUIT"),
        R"DIAGRAM(
        q0: -@-@-
             | |
        q1: -X-|-
             | |
        q2: -X-X-
             | |
        q3: -X-|-
             | |
        q4: -X-X-
    )DIAGRAM");

    expect_circuit_has_text_diagram(
        Circuit(R"CIRCUIT(
            APPEND_TO_REGISTER q1 r0
            APPEND_TO_REGISTER q2 r0
            APPEND_TO_REGISTER q3 r0
            APPEND_TO_REGISTER q4 r0
            APPEND_TO_REGISTER q5 r0
            CX q0 q3
            CX q0 q4
            CX q0 q5
            CZ q0 q1
            CZ q0 q2
            CZ q0 q3
        )CIRCUIT"),
        R"DIAGRAM(
        q0: ---------@-@-
                     | |
        q1: -reg0[0]-Z-|-
                     | |
        q2: -reg0[1]-Z-|-
                     | |
        q3: -reg0[2]-X-Z-
                     |
        q4: -reg0[3]-X---
                     |
        q5: -reg0[4]-X---
    )DIAGRAM");
}

TEST(circuit, text_diagram_supports_all_gates) {
    Circuit circuit = circuit_with_every_operation();
    ASSERT_NE(circuit.text_diagram(), "");
}

TEST(circuit, text_diagram_z_pow) {
    expect_circuit_has_text_diagram(
        Circuit(R"CIRCUIT(
            Z_POW q0 0.25
            Z_POW q1 -0.5 if b0
        )CIRCUIT"),
        R"DIAGRAM(
            q0: -Z^0.25--------
                        if(b0)
            q1: --------Z^1.5--
    )DIAGRAM");
}

TEST(circuit, max_magic) {
    ASSERT_EQ(
        Circuit(R"CIRCUIT(
        )CIRCUIT")
            .max_magic(),
        0);
    ASSERT_EQ(
        Circuit(R"CIRCUIT(
            CCX q0 q1 q2
            CCZ q0 q1 q2
            CCX q0 q1 q2 if b0
            CCZ q0 q1 q2 if b1
        )CIRCUIT")
            .max_magic(),
        4);
    ASSERT_EQ(
        Circuit(R"CIRCUIT(
        REGISTER r0
        APPEND_TO_REGISTER q0 r0
        APPEND_TO_REGISTER b0 r0
        DEBUG_PRINT
        X q0
        X q1 if b0
        CX q0 q1
        CX q1 q2 if b1
        NEG
        NEG if b3
        Z q5
        Z q6 if b4
        CZ q5 q6
        CZ q6 q7 if b5
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
        PUSH_CONDITION if b21
        SWAP q13 q14
        SWAP q15 q16 if b20
        POP_CONDITION
    )CIRCUIT")
            .max_magic(),
        0);
}

TEST(circuit, reaction_depth) {
    ASSERT_EQ(
        Circuit(R"CIRCUIT(
    )CIRCUIT")
            .reaction_depth(),
        0);

    ASSERT_EQ(
        Circuit(R"CIRCUIT(
        X q0
    )CIRCUIT")
            .reaction_depth(),
        0);

    ASSERT_EQ(
        Circuit(R"CIRCUIT(
        CX q0 q1
    )CIRCUIT")
            .reaction_depth(),
        0);

    ASSERT_EQ(
        Circuit(R"CIRCUIT(
        CCX q0 q1 q2
    )CIRCUIT")
            .reaction_depth(),
        1);

    ASSERT_EQ(
        Circuit(R"CIRCUIT(
        HMR q0 b0
    )CIRCUIT")
            .reaction_depth(),
        1);

    ASSERT_EQ(
        Circuit(R"CIRCUIT(
        HMR q0 b0
        X q1 if b0
    )CIRCUIT")
            .reaction_depth(),
        1);

    ASSERT_EQ(
        Circuit(R"CIRCUIT(
        HMR q0 b0
        HMR q1 b1
        X q2 if b0
        X q2 if b1
    )CIRCUIT")
            .reaction_depth(),
        1);

    ASSERT_EQ(
        Circuit(R"CIRCUIT(
        HMR q0 b0
        X q1 if b0
        HMR q1 b1
        X q2 if b1
    )CIRCUIT")
            .reaction_depth(),
        1);

    ASSERT_EQ(
        Circuit(R"CIRCUIT(
        HMR q0 b0
        Z q1 if b0
        HMR q1 b1
        X q2 if b1
    )CIRCUIT")
            .reaction_depth(),
        2);

    ASSERT_EQ(
        Circuit(R"CIRCUIT(
        CCX q0 q1 q2
        CCX q4 q3 q2
    )CIRCUIT")
            .reaction_depth(),
        1);

    ASSERT_EQ(
        Circuit(R"CIRCUIT(
        CCX q0 q1 q2
        CCX q2 q3 q4
    )CIRCUIT")
            .reaction_depth(),
        2);

    ASSERT_EQ(
        Circuit(R"CIRCUIT(
        CCX q2 q3 q4
        CCX q0 q1 q2
    )CIRCUIT")
            .reaction_depth(),
        2);

    ASSERT_EQ(
        Circuit(R"CIRCUIT(
        CCX q0 q1 q2
        CCX q2 q3 q4
        CCX q0 q1 q2
        CCX q2 q3 q4
        CCX q0 q1 q2
        CCX q2 q3 q4
        CCX q0 q1 q2
        CCX q2 q3 q4
    )CIRCUIT")
            .reaction_depth(),
        2);

    ASSERT_EQ(
        Circuit(R"CIRCUIT(
        CX q7 q3
        R q8
        CCX q0 q4 q8
        CX q8 q5
        CX q8 q1
        R q9
        CCX q1 q5 q9
        CX q9 q8
        CX q8 q6
        CX q8 q2
        CX q8 q3
        CCX q2 q6 q3
        CX q8 q6
        CX q6 q2
        CX q9 q8
        HMR q9 b0
        CZ q1 q5 if b0
        CX q8 q5
        CX q5 q1
        HMR q8 b0
        CZ q0 q4 if b0
        CX q4 q0
    )CIRCUIT")
            .reaction_depth(),
        5);

    ASSERT_EQ(
        Circuit(R"CIRCUIT(
            CCZ q0 q1 q2
            CCZ q0 q1 q2
        )CIRCUIT")
            .reaction_depth(),
        1);

    ASSERT_EQ(
        Circuit(R"CIRCUIT(
            Z_POW q0 0.25
        )CIRCUIT")
            .reaction_depth(),
        1);

    ASSERT_EQ(
        Circuit(R"CIRCUIT(
            Z_POW q0 0.25
            Z_POW q0 0.25
        )CIRCUIT")
            .reaction_depth(),
        1);

    ASSERT_EQ(
        Circuit(R"CIRCUIT(
            Z_POW q2 0.25
            CCZ q0 q1 q2
            Z_POW q2 0.25
        )CIRCUIT")
            .reaction_depth(),
        1);

    ASSERT_EQ(
        Circuit(R"CIRCUIT(
            Z_POW q2 0.25
            CCX q0 q1 q2
            Z_POW q2 0.25
        )CIRCUIT")
            .reaction_depth(),
        2);

    ASSERT_EQ(
        Circuit(R"CIRCUIT(
            CCX q0 q1 q2
            Z_POW q2 0.25
            CCX q0 q1 q2
            Z_POW q2 0.25
        )CIRCUIT")
            .reaction_depth(),
        3);

    ASSERT_EQ(
        Circuit(R"CIRCUIT(
            CCX q0 q1 q2
            Z_POW q0 0.25
            CCX q0 q1 q2
            Z_POW q0 0.25
        )CIRCUIT")
            .reaction_depth(),
        1);
}

TEST(circuit, write_to) {
    Circuit e = circuit_with_every_operation();
    FILE *f = tmpfile();
    e.write_kmx_to(f);
    fseek(f, 0, SEEK_SET);
    std::string contents;
    char buf[1024];
    while (size_t n = fread(buf, 1, 1024, f)) {
        for (size_t k = 0; k < n; k++) {
            contents.push_back(buf[k]);
        }
    }
    fclose(f);
    ASSERT_EQ(contents, e.str() + "\n");
}

TEST(circuit, kmb_every_operation) {
    RaiiTempNamedFile kmb;
    Circuit circuit = circuit_with_every_operation();
    FILE *f = fopen(kmb.path.c_str(), "wb");
    circuit.write_kmb_to(f);
    fclose(f);
    f = fopen(kmb.path.c_str(), "rb");
    Circuit restored = Circuit::from_kmb_file(f);
    ASSERT_EQ(restored, circuit);
}

static std::string bytestr(const std::vector<uint8_t> &v) {
    std::string out;
    for (auto e : v) {
        if (!out.empty()) {
            out.push_back(' ');
        }
        if (e == 0) {
            out.append(" .");
        } else {
            out.push_back(" 123456789abcdef"[e / 16]);
            out.push_back("0123456789abcdef"[e % 16]);
        }
    }
    return out;
}
static std::vector<uint8_t> bytestr(const char *text) {
    std::vector<uint8_t> result;
    while (true) {
        while (*text == ' ' || *text == '\n') {
            text++;
        }
        if (*text == '#') {
            while (*text != '\n') {
                text++;
            }
            continue;
        }
        if (*text == '\0') {
            break;
        }
        const char *start = text;
        while (*text != ' ' && *text != '\0' && *text != '\n') {
            text++;
        }
        std::string_view term(start, text);
        if (term == "." || term == "..") {
            result.push_back(0);
            continue;
        }
        if (term.size() == 1 || term.size() == 2) {
            uint8_t v = 0;
            for (char c : term) {
                v <<= 4;
                if (c >= '0' && c <= '9') {
                    v += c - '0';
                } else if (c >= 'a' && c <= 'f') {
                    v += 10 + (c - 'a');
                } else if (c >= 'A' && c <= 'f') {
                    v += 10 + (c - 'A');
                } else {
                    throw std::invalid_argument(std::string(term));
                }
            }
            result.push_back(v);
            continue;
        }
        throw std::invalid_argument(std::string(term));
    }
    return result;
}

void expect_circuit_serializes_to_kmb(std::string_view circuit_text, const char *byte_text) {
    RaiiTempNamedFile kmb;
    Circuit circuit(circuit_text);
    FILE *f = fopen(kmb.path.c_str(), "wb");
    circuit.write_kmb_to(f);
    fclose(f);
    auto actual = kmb.read_bytes();
    auto expected = bytestr(byte_text);
    if (actual != expected) {
        EXPECT_EQ(actual, expected) << "\n    actual: " << bytestr(actual) << "\n    expect: " << bytestr(expected);
    }
}

TEST(circuit, kmb_empty) {
    expect_circuit_serializes_to_kmb("", R"HEX(
        d7 50 c7 d5 c3 29 d3 26 e3 cc 9f 68 34 f2 b8 bf
         2  .  .  .  # version
         0  .  .  .  # num_qubits
         0  .  .  .  # num_bits
         0  .  .  .  # num_reg
         0  .  .  .  .  .  .  .  # num_ops
         0  .  .  .  .  .  .  .  # num_qqq_ops
         0  .  .  .  .  .  .  .  # num_qq_ops
         0  .  .  .  .  .  .  .  # num_q_ops
         0  .  .  .  .  .  .  .  # num_b_ops
         0  .  .  .  .  .  .  .  # num_c_ops
         0  .  .  .  .  .  .  .  # num_angle_ops
         1  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # op types
         2  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq0s
         3  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq1s
         4  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq2s
         5  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq0s
         6  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq1s
         7  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # q0s
         8  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # b0s
         9  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # bcs
         A  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # angles
    )HEX");
}

TEST(circuit, kmb_x) {
    expect_circuit_serializes_to_kmb(
        R"CIRCUIT(
        X q14
    )CIRCUIT",
        R"HEX(
        d7 50 c7 d5 c3 29 d3 26 e3 cc 9f 68 34 f2 b8 bf
         2  .  .  .  # version
         f  .  .  .  # num_qubits
         0  .  .  .  # num_bits
         0  .  .  .  # num_reg
         1  .  .  .  .  .  .  .  # num_ops
         0  .  .  .  .  .  .  .  # num_qqq_ops
         0  .  .  .  .  .  .  .  # num_qq_ops
         1  .  .  .  .  .  .  .  # num_q_ops
         0  .  .  .  .  .  .  .  # num_b_ops
         0  .  .  .  .  .  .  .  # num_c_ops
         0  .  .  .  .  .  .  .  # num_angles
         1  .  .  .  .  .  .  .  1  .  .  .  .  .  .  .  # op types
        11
         2  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq0s
         3  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq1s
         4  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq2s
         5  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq0s
         6  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq1s
         7  .  .  .  .  .  .  .  4  .  .  .  .  .  .  .  # q0s
         e  .  .  .
         8  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # b0s
         9  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # bcs
         A  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # angles
    )HEX");
}

TEST(circuit, kmb_cx) {
    expect_circuit_serializes_to_kmb(
        R"CIRCUIT(
        CX q14 q19
    )CIRCUIT",
        R"HEX(
        d7 50 c7 d5 c3 29 d3 26 e3 cc 9f 68 34 f2 b8 bf
         2  .  .  .  # version
        14  .  .  .  # num_qubits
         0  .  .  .  # num_bits
         0  .  .  .  # num_reg
         1  .  .  .  .  .  .  .  # num_ops
         0  .  .  .  .  .  .  .  # num_qqq_ops
         1  .  .  .  .  .  .  .  # num_qq_ops
         0  .  .  .  .  .  .  .  # num_q_ops
         0  .  .  .  .  .  .  .  # num_b_ops
         0  .  .  .  .  .  .  .  # num_c_ops
         0  .  .  .  .  .  .  .  # num_angles
         1  .  .  .  .  .  .  .  1  .  .  .  .  .  .  .  # op types
        21
         2  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq0s
         3  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq1s
         4  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq2s
         5  .  .  .  .  .  .  .  4  .  .  .  .  .  .  .  # qq0s
         e  .  .  .
         6  .  .  .  .  .  .  .  4  .  .  .  .  .  .  .  # qq1s
        13  .  .  .
         7  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # q0s
         8  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # b0s
         9  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # bcs
         A  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # angles
    )HEX");
}

TEST(circuit, kmb_ccz_if) {
    expect_circuit_serializes_to_kmb(
        R"CIRCUIT(
        CCZ q14 q19 q21 if b5
    )CIRCUIT",
        R"HEX(
        d7 50 c7 d5 c3 29 d3 26 e3 cc 9f 68 34 f2 b8 bf
         2  .  .  .  # version
        16  .  .  .  # num_qubits
         6  .  .  .  # num_bits
         0  .  .  .  # num_reg
         1  .  .  .  .  .  .  .  # num_ops
         1  .  .  .  .  .  .  .  # num_qqq_ops
         0  .  .  .  .  .  .  .  # num_qq_ops
         0  .  .  .  .  .  .  .  # num_q_ops
         0  .  .  .  .  .  .  .  # num_b_ops
         1  .  .  .  .  .  .  .  # num_c_ops
         0  .  .  .  .  .  .  .  # num_angles
         1  .  .  .  .  .  .  .  1  .  .  .  .  .  .  .  # op types
        b0
         2  .  .  .  .  .  .  .  4  .  .  .  .  .  .  .  # qqq0s
         e  .  .  .
         3  .  .  .  .  .  .  .  4  .  .  .  .  .  .  .  # qqq1s
        13  .  .  .
         4  .  .  .  .  .  .  .  4  .  .  .  .  .  .  .  # qqq2s
        15  .  .  .
         5  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq0s
         6  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq1s
         7  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # q0s
         8  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # b0s
         9  .  .  .  .  .  .  .  4  .  .  .  .  .  .  .  # bcs
         5  .  .  .
         A  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # angles
    )HEX");
}

TEST(circuit, kmb_registers) {
    expect_circuit_serializes_to_kmb(
        R"CIRCUIT(
        APPEND_TO_REGISTER q0 r0
        APPEND_TO_REGISTER q1 r0
        APPEND_TO_REGISTER b2 r1
        REGISTER r0 "test"
        HMR q2 b3
        BIT_STORE0 b4
    )CIRCUIT",
        R"HEX(
        d7 50 c7 d5 c3 29 d3 26 e3 cc 9f 68 34 f2 b8 bf
         2  .  .  .  # version
         3  .  .  .  # num_qubits
         5  .  .  .  # num_bits
         2  .  .  .  # num_reg
         2  .  .  .  .  .  .  .  # num_ops
         0  .  .  .  .  .  .  .  # num_qqq_ops
         0  .  .  .  .  .  .  .  # num_qq_ops
         1  .  .  .  .  .  .  .  # num_q_ops
         2  .  .  .  .  .  .  .  # num_b_ops
         0  .  .  .  .  .  .  .  # num_c_ops
         0  .  .  .  .  .  .  .  # num_angles
         0  .  .  .  .  .  .  . 14  .  .  .  .  .  .  .  # register0 data
         4  .  .  .
         2  .  .  .
        74 65 73 74  # test
         0  .  . 80
         1  .  . 80
         0  .  .  .  .  .  .  .  C  .  .  .  .  .  .  .  # register1 data
         0  .  .  .
         1  .  .  .
         2  .  .  .
         1  .  .  .  .  .  .  .  2  .  .  .  .  .  .  .  # op types
        50 40
         2  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq0s
         3  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq1s
         4  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq2s
         5  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq0s
         6  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq1s
         7  .  .  .  .  .  .  .  4  .  .  .  .  .  .  .  # q0s
         2  .  .  .
         8  .  .  .  .  .  .  .  8  .  .  .  .  .  .  .  # b0s
         3  .  .  .
         4  .  .  .
         9  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # bcs
         A  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # angles
    )HEX");
}

TEST(circuit, kmb_circuit_with_every_operation) {
    expect_circuit_serializes_to_kmb(
        R"CIRCUIT(
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

        Z_POW q8 0.89286214436147007517525473026169148818015249963907853147982813801624058773023933057908473054364861809517606161534786224365234375
        Z_POW q7 0.03736581851516919846777389081721871603498536409602155139062903858467442707455202086032526598291525488093611784279346466064453125 if b5
    )CIRCUIT",
        R"HEX(
        d7 50 c7 d5 c3 29 d3 26 e3 cc 9f 68 34 f2 b8 bf
         2  .  .  .  # version
        16  .  .  .  # num_qubits
        19  .  .  .  # num_bits
         2  .  .  .  # num_reg
        25  .  .  .  .  .  .  .  # num_ops
         4  .  .  .  .  .  .  .  # num_qqq_ops
         6  .  .  .  .  .  .  .  # num_qq_ops
         c  .  .  .  .  .  .  .  # num_q_ops
         a  .  .  .  .  .  .  .  # num_b_ops
        10  .  .  .  .  .  .  .  # num_c_ops
         2  .  .  .  .  .  .  .  # num_angles
         0  .  .  .  .  .  .  . 10  .  .  .  .  .  .  .  # register0 data
         0  .  .  .
         2  .  .  .
         0  .  . 80
         0  .  .  .
         0  .  .  .  .  .  .  . 11  .  .  .  .  .  .  .  # register1 data
         9  .  .  .
         0  .  .  .
        74 65 73 74 0d 0a 5c 22 23
         1  .  .  .  .  .  .  . 25  .  .  .  .  .  .  .  # op types
               07 11 91 21 a1 31 b1 00 80 10 90 20 a0 30 b0 50 d0 12 92 42 c2 40 c0 41 c1 22 a2 81 41 11 81 41 11 01 01 13 93
         2  .  .  .  .  .  .  .  10  .  .  .  .  .  .  .  # qqq0s
               0  .  .  .
               1  .  .  .
               6  .  .  .
               7  .  .  .
         3  .  .  .  .  .  .  . 10  .  .  . .  .  .  .  # qqq1s
               1  .  .  .
               2  .  .  .
               7  .  .  .
               8  .  .  .
         4  .  .  .  .  .  .  . 10  .  .  . .  .  .  .  # qqq2s
               2  .  .  .
               3  .  .  .
               8  .  .  .
               9  .  .  .
         5  .  .  .  .  .  .  . 18  .  .  .  .  .  .  .  # qq0s
               0  .  .  .
               1  .  .  .
               5  .  .  .
               6  .  .  .
               d  .  .  .
               f  .  .  .
         6  .  .  .  .  .  .  . 18  .  .  .  .  .  .  .  # qq1s
               1  .  .  .
               2  .  .  .
               6  .  .  .
               7  .  .  .
               e  .  .  .
              10  .  .  .
         7  .  .  .  .  .  .  . 30  .  .  .  .  .  .  .  # q0s
               0  .  .  .
               1  .  .  .
               5  .  .  .
               6  .  .  .
               9  .  .  .
               a  .  .  .
               b  .  .  .
               c  .  .  .
              14  .  .  .
              15  .  .  .
               8  .  .  .
               7  .  .  .
         8  .  .  .  .  .  .  . 28  .  .  .  .  .  .  .  # b0s
               7  .  .  .
              11  .  .  .
               8  .  .  .
               9  .  .  .
               b  .  .  .
               c  .  .  .
               e  .  .  .
               f  .  .  .
              17  .  .  .
              18  .  .  .
         9  .  .  .  .  .  .  . 40  .  .  .  .  .  .  .  # bcs
               0  .  .  .
               1  .  .  .
               2  .  .  .
               3  .  .  .
               4  .  .  .
               5  .  .  .
               6  .  .  .
              12  .  .  .
              13  .  .  .
               a  .  .  .
               d  .  .  .
              10  .  .  .
              14  .  .  .
              15  .  .  .
              16  .  .  .
               5  .  .  .
         A  .  .  .  .  .  .  . 20  .  .  .  .  .  .  .  # angles
               64 5b ce c7  c a9 f8 19 18 95 39 ef 86 4e 49 72
                2  c b7 75 47 25 49 ab a8 ec 65 41 34 67 c8  4
    )HEX");
}

void expect_fails_to_parse(const char *substring, const char *byte_text) {
    RaiiTempNamedFile kmb;
    FILE *f = fopen(kmb.path.c_str(), "wb");
    auto data = bytestr(byte_text);
    if (fwrite(data.data(), data.size(), 1, f) != 1) {
        throw std::invalid_argument("Failed to write test data");
    }
    fclose(f);
    f = fopen(kmb.path.c_str(), "rb");
    try {
        Circuit::from_kmb_file(f);
    } catch (const std::invalid_argument &ex) {
        fclose(f);
        auto v = std::string_view(ex.what());
        if (v.find(substring) == std::string_view::npos) {
            EXPECT_FALSE(true) << "'" << substring << "' not in '" << ex.what() << "'";
        }
        return;
    }
    fclose(f);
    EXPECT_FALSE(true) << "Didn't raise an exception (and so it didn't contain '" << substring << "')";
}

TEST(circuit, kmb_validate_unknown_version) {
    expect_fails_to_parse("file version", R"HEX(
        d7 50 c7 d5 c3 29 d3 26 e3 cc 9f 68 34 f2 b8 bf
         1  .  .  .  # version
         0  .  .  .  # num_qubits
         0  .  .  .  # num_bits
         0  .  .  .  # num_reg
         0  .  .  .  .  .  .  .  # num_ops
         0  .  .  .  .  .  .  .  # num_qqq_ops
         0  .  .  .  .  .  .  .  # num_qq_ops
         0  .  .  .  .  .  .  .  # num_q_ops
         0  .  .  .  .  .  .  .  # num_b_ops
         0  .  .  .  .  .  .  .  # num_c_ops
         0  .  .  .  .  .  .  .  # num_angles
         1  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # op types
         2  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq0s
         3  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq1s
         4  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq2s
         5  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq0s
         6  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq1s
         7  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # q0s
         8  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # b0s
         9  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # bcs
         A  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # angles
    )HEX");
}

TEST(circuit, kmb_validate_num_qubits) {
    expect_fails_to_parse("num_qubits", R"HEX(
        d7 50 c7 d5 c3 29 d3 26 e3 cc 9f 68 34 f2 b8 bf
         2  .  .  .  # version
         1  .  .  .  # num_qubits
         0  .  .  .  # num_bits
         0  .  .  .  # num_reg
         0  .  .  .  .  .  .  .  # num_ops
         0  .  .  .  .  .  .  .  # num_qqq_ops
         0  .  .  .  .  .  .  .  # num_qq_ops
         0  .  .  .  .  .  .  .  # num_q_ops
         0  .  .  .  .  .  .  .  # num_b_ops
         0  .  .  .  .  .  .  .  # num_c_ops
         0  .  .  .  .  .  .  .  # num_angles
         1  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # op types
         2  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq0s
         3  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq1s
         4  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq2s
         5  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq0s
         6  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq1s
         7  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # q0s
         8  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # b0s
         9  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # bcs
         A  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # angles
    )HEX");
}

TEST(circuit, kmb_validate_num_bits) {
    expect_fails_to_parse("num_bits", R"HEX(
        d7 50 c7 d5 c3 29 d3 26 e3 cc 9f 68 34 f2 b8 bf
         2  .  .  .  # version
         0  .  .  .  # num_qubits
         1  .  .  .  # num_bits
         0  .  .  .  # num_reg
         0  .  .  .  .  .  .  .  # num_ops
         0  .  .  .  .  .  .  .  # num_qqq_ops
         0  .  .  .  .  .  .  .  # num_qq_ops
         0  .  .  .  .  .  .  .  # num_q_ops
         0  .  .  .  .  .  .  .  # num_b_ops
         0  .  .  .  .  .  .  .  # num_c_ops
         0  .  .  .  .  .  .  .  # num_angles
         1  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # op types
         2  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq0s
         3  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq1s
         4  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq2s
         5  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq0s
         6  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq1s
         7  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # q0s
         8  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # b0s
         9  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # bcs
         A  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # angles
    )HEX");
}

TEST(circuit, kmb_validate_num_registers) {
    expect_fails_to_parse("Expected register target data", R"HEX(
        d7 50 c7 d5 c3 29 d3 26 e3 cc 9f 68 34 f2 b8 bf
         2  .  .  .  # version
         0  .  .  .  # num_qubits
         0  .  .  .  # num_bits
         1  .  .  .  # num_reg
         0  .  .  .  .  .  .  .  # num_ops
         0  .  .  .  .  .  .  .  # num_qqq_ops
         0  .  .  .  .  .  .  .  # num_qq_ops
         0  .  .  .  .  .  .  .  # num_q_ops
         0  .  .  .  .  .  .  .  # num_b_ops
         0  .  .  .  .  .  .  .  # num_c_ops
         0  .  .  .  .  .  .  .  # num_angles
         1  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # op types
         2  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq0s
         3  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq1s
         4  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq2s
         5  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq0s
         6  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq1s
         7  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # q0s
         8  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # b0s
         9  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # bcs
         A  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # angles
    )HEX");
}

TEST(circuit, kmb_validate_num_registers2) {
    expect_fails_to_parse("Expected operation type data", R"HEX(
        d7 50 c7 d5 c3 29 d3 26 e3 cc 9f 68 34 f2 b8 bf
         2  .  .  .  # version
         0  .  .  .  # num_qubits
         0  .  .  .  # num_bits
         1  .  .  .  # num_reg
         0  .  .  .  .  .  .  .  # num_ops
         0  .  .  .  .  .  .  .  # num_qqq_ops
         0  .  .  .  .  .  .  .  # num_qq_ops
         0  .  .  .  .  .  .  .  # num_q_ops
         0  .  .  .  .  .  .  .  # num_b_ops
         0  .  .  .  .  .  .  .  # num_c_ops
         0  .  .  .  .  .  .  .  # num_angles
         0  .  .  .  .  .  .  .  8  .  .  .  .  .  .  .  # register data 0
         0  .  .  .
         0  .  .  .
         0  .  .  .  .  .  .  .  8  .  .  .  .  .  .  .  # register data 1
         0  .  .  .
         0  .  .  .
         1  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # op types
         2  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq0s
         3  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq1s
         4  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq2s
         5  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq0s
         6  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq1s
         7  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # q0s
         8  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # b0s
         9  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # bcs
         A  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # angles
    )HEX");
}

TEST(circuit, kmb_validate_num_ops) {
    expect_fails_to_parse("type payload size", R"HEX(
        d7 50 c7 d5 c3 29 d3 26 e3 cc 9f 68 34 f2 b8 bf
         2  .  .  .  # version
         0  .  .  .  # num_qubits
         0  .  .  .  # num_bits
         0  .  .  .  # num_reg
         1  .  .  .  .  .  .  .  # num_ops
         0  .  .  .  .  .  .  .  # num_qqq_ops
         0  .  .  .  .  .  .  .  # num_qq_ops
         0  .  .  .  .  .  .  .  # num_q_ops
         0  .  .  .  .  .  .  .  # num_b_ops
         0  .  .  .  .  .  .  .  # num_c_ops
         0  .  .  .  .  .  .  .  # num_angles
         1  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # op types
         2  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq0s
         3  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq1s
         4  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq2s
         5  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq0s
         6  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq1s
         7  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # q0s
         8  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # b0s
         9  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # bcs
         A  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # angles
    )HEX");
}

TEST(circuit, kmb_validate_num_qqq_ops) {
    expect_fails_to_parse("actual payload size vs size specified in header", R"HEX(
        d7 50 c7 d5 c3 29 d3 26 e3 cc 9f 68 34 f2 b8 bf
         2  .  .  .  # version
         0  .  .  .  # num_qubits
         0  .  .  .  # num_bits
         0  .  .  .  # num_reg
         0  .  .  .  .  .  .  .  # num_ops
         1  .  .  .  .  .  .  .  # num_qqq_ops
         0  .  .  .  .  .  .  .  # num_qq_ops
         0  .  .  .  .  .  .  .  # num_q_ops
         0  .  .  .  .  .  .  .  # num_b_ops
         0  .  .  .  .  .  .  .  # num_c_ops
         0  .  .  .  .  .  .  .  # num_angles
         1  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # op types
         2  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq0s
         3  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq1s
         4  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq2s
         5  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq0s
         6  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq1s
         7  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # q0s
         8  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # b0s
         9  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # bcs
         A  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # angles
    )HEX");
}

TEST(circuit, kmb_validate_num_qq_ops) {
    expect_fails_to_parse("actual payload size vs size specified in header", R"HEX(
        d7 50 c7 d5 c3 29 d3 26 e3 cc 9f 68 34 f2 b8 bf
         2  .  .  .  # version
         0  .  .  .  # num_qubits
         0  .  .  .  # num_bits
         0  .  .  .  # num_reg
         0  .  .  .  .  .  .  .  # num_ops
         0  .  .  .  .  .  .  .  # num_qqq_ops
         1  .  .  .  .  .  .  .  # num_qq_ops
         0  .  .  .  .  .  .  .  # num_q_ops
         0  .  .  .  .  .  .  .  # num_b_ops
         0  .  .  .  .  .  .  .  # num_c_ops
         0  .  .  .  .  .  .  .  # num_angles
         1  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # op types
         2  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq0s
         3  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq1s
         4  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq2s
         5  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq0s
         6  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq1s
         7  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # q0s
         8  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # b0s
         9  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # bcs
         A  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # angles
    )HEX");
}

TEST(circuit, kmb_validate_num_q_ops) {
    expect_fails_to_parse("actual payload size vs size specified in header", R"HEX(
        d7 50 c7 d5 c3 29 d3 26 e3 cc 9f 68 34 f2 b8 bf
         2  .  .  .  # version
         0  .  .  .  # num_qubits
         0  .  .  .  # num_bits
         0  .  .  .  # num_reg
         0  .  .  .  .  .  .  .  # num_ops
         0  .  .  .  .  .  .  .  # num_qqq_ops
         0  .  .  .  .  .  .  .  # num_qq_ops
         1  .  .  .  .  .  .  .  # num_q_ops
         0  .  .  .  .  .  .  .  # num_b_ops
         0  .  .  .  .  .  .  .  # num_c_ops
         0  .  .  .  .  .  .  .  # num_angles
         1  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # op types
         2  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq0s
         3  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq1s
         4  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq2s
         5  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq0s
         6  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq1s
         7  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # q0s
         8  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # b0s
         9  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # bcs
         A  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # angles
    )HEX");
}

TEST(circuit, kmb_validate_num_b_ops) {
    expect_fails_to_parse("actual payload size vs size specified in header", R"HEX(
        d7 50 c7 d5 c3 29 d3 26 e3 cc 9f 68 34 f2 b8 bf
         2  .  .  .  # version
         0  .  .  .  # num_qubits
         0  .  .  .  # num_bits
         0  .  .  .  # num_reg
         0  .  .  .  .  .  .  .  # num_ops
         0  .  .  .  .  .  .  .  # num_qqq_ops
         0  .  .  .  .  .  .  .  # num_qq_ops
         0  .  .  .  .  .  .  .  # num_q_ops
         1  .  .  .  .  .  .  .  # num_b_ops
         0  .  .  .  .  .  .  .  # num_c_ops
         0  .  .  .  .  .  .  .  # num_angles
         1  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # op types
         2  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq0s
         3  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq1s
         4  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq2s
         5  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq0s
         6  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq1s
         7  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # q0s
         8  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # b0s
         9  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # bcs
         A  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # angles
    )HEX");
}

TEST(circuit, kmb_validate_num_c_ops) {
    expect_fails_to_parse("actual payload size vs size specified in header", R"HEX(
        d7 50 c7 d5 c3 29 d3 26 e3 cc 9f 68 34 f2 b8 bf
         2  .  .  .  # version
         0  .  .  .  # num_qubits
         0  .  .  .  # num_bits
         0  .  .  .  # num_reg
         0  .  .  .  .  .  .  .  # num_ops
         0  .  .  .  .  .  .  .  # num_qqq_ops
         0  .  .  .  .  .  .  .  # num_qq_ops
         0  .  .  .  .  .  .  .  # num_q_ops
         0  .  .  .  .  .  .  .  # num_b_ops
         1  .  .  .  .  .  .  .  # num_c_ops
         0  .  .  .  .  .  .  .  # num_angles
         1  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # op types
         2  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq0s
         3  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq1s
         4  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq2s
         5  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq0s
         6  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq1s
         7  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # q0s
         8  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # b0s
         9  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # bcs
         A  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # angles
    )HEX");
}

TEST(circuit, kmb_validate_magic_bytes) {
    expect_fails_to_parse("magic bytes", R"HEX(
        d7 50 c7 d5 c3 29 d4 26 e3 cc 9f 68 34 f2 b8 bf
         2  .  .  .  # version
         0  .  .  .  # num_qubits
         0  .  .  .  # num_bits
         0  .  .  .  # num_reg
         0  .  .  .  .  .  .  .  # num_ops
         0  .  .  .  .  .  .  .  # num_qqq_ops
         0  .  .  .  .  .  .  .  # num_qq_ops
         0  .  .  .  .  .  .  .  # num_q_ops
         0  .  .  .  .  .  .  .  # num_b_ops
         0  .  .  .  .  .  .  .  # num_c_ops
         0  .  .  .  .  .  .  .  # num_angles
         1  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # op types
         2  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq0s
         3  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq1s
         4  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qqq2s
         5  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq0s
         6  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq1s
         7  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # q0s
         8  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # b0s
         9  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # bcs
         A  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # angles
    )HEX");
}

TEST(circuit, kmb_validate_non_zero_num_qqq) {
    expect_fails_to_parse("num_qqq_ops=1 != compute_num_qqq()=0", R"HEX(
        d7 50 c7 d5 c3 29 d3 26 e3 cc 9f 68 34 f2 b8 bf
         2  .  .  .  # version
         3  .  .  .  # num_qubits
         0  .  .  .  # num_bits
         0  .  .  .  # num_reg
         1  .  .  .  .  .  .  .  # num_ops
         1  .  .  .  .  .  .  .  # num_qqq_ops
         0  .  .  .  .  .  .  .  # num_qq_ops
         0  .  .  .  .  .  .  .  # num_q_ops
         0  .  .  .  .  .  .  .  # num_b_ops
         0  .  .  .  .  .  .  .  # num_c_ops
         0  .  .  .  .  .  .  .  # num_angles
         1  .  .  .  .  .  .  .  1  .  .  .  .  .  .  .  # op types
                 00
         2  .  .  .  .  .  .  .  4  .  .  .  .  .  .  .  # qqq0s
                  0  .  .  .
         3  .  .  .  .  .  .  .  4  .  .  .  .  .  .  .  # qqq1s
                  1  .  .  .
         4  .  .  .  .  .  .  .  4  .  .  .  .  .  .  .  # qqq2s
                  2  .  .  .
         5  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq0s
         6  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # qq1s
         7  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # q0s
         8  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # b0s
         9  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # bcs
         A  .  .  .  .  .  .  .  0  .  .  .  .  .  .  .  # angles
    )HEX");
}

TEST(circuit, kmx_file_and_stream_agree) {
    Circuit c = circuit_with_every_operation();
    RaiiTempNamedFile tmp;
    FILE *f = fopen(tmp.path.c_str(), "wb");
    c.write_kmx_to(f);
    fclose(f);
    auto file_contents = tmp.read_contents();
    std::stringstream ss;
    c.write_kmx_to(ss);
    auto stream_contents = ss.str();
    while (file_contents.ends_with('\n')) {
        file_contents.pop_back();
    }
    while (stream_contents.ends_with('\n')) {
        stream_contents.pop_back();
    }
    ASSERT_EQ(file_contents, stream_contents);
}

TEST(circuit, named_registers) {
    Circuit c(R"CIRCUIT(
        REGISTER r0 "offset"
        APPEND_TO_REGISTER q1 r0
        APPEND_TO_REGISTER b2 r1
        REGISTER r1 "target"
    )CIRCUIT");
    ASSERT_EQ(
        c.register_data,
        (std::vector<RegisterData>{
            {.name = "offset", .contents = {QubitId(1)}},
            {.name = "target", .contents = {BitId(2)}},
        }));

    c = Circuit(R"CIRCUIT(
        REGISTER r0 "offset"
        REGISTER r0 "overwrite"
    )CIRCUIT");
    ASSERT_EQ(
        c.register_data,
        (std::vector<RegisterData>{
            {.name = "overwrite", .contents = {}},
        }));

    c = Circuit(R"CIRCUIT(
        REGISTER r0 "\r\n\P\Q\B"
    )CIRCUIT");
    ASSERT_EQ(
        c.register_data,
        (std::vector<RegisterData>{
            {.name = "\r\n#\"\\", .contents = {}},
        }));
    ASSERT_EQ(c.str(), R"CIRCUIT(REGISTER r0 "\r\n\P\Q\B")CIRCUIT");

    c = Circuit(R"CIRCUIT(
        REGISTER r0 ""
    )CIRCUIT");
    ASSERT_EQ(
        c.register_data,
        (std::vector<RegisterData>{
            {.name = "", .contents = {}},
        }));

    c = Circuit(R"CIRCUIT(
        REGISTER r0
    )CIRCUIT");
    ASSERT_EQ(
        c.register_data,
        (std::vector<RegisterData>{
            {.name = "", .contents = {}},
        }));

    c = Circuit(R"CIRCUIT(
        REGISTER r0  # "offset"
    )CIRCUIT");
    ASSERT_EQ(
        c.register_data,
        (std::vector<RegisterData>{
            {.name = "", .contents = {}},
        }));

    EXPECT_THROW(
        {
            Circuit(R"CIRCUIT(
            REGISTER r0 "offset
        )CIRCUIT");
        },
        std::invalid_argument);

    EXPECT_THROW(
        {
            Circuit(R"CIRCUIT(
            REGISTER r0 offset
        )CIRCUIT");
        },
        std::invalid_argument);

    EXPECT_THROW(
        {
            Circuit(R"CIRCUIT(
            REGISTER r0 "offset#" #test
        )CIRCUIT");
        },
        std::invalid_argument);

    EXPECT_THROW(
        {
            Circuit(R"CIRCUIT(
            REGISTER r0 "offset\m"
        )CIRCUIT");
        },
        std::invalid_argument);

    EXPECT_THROW(
        {
            Circuit(R"CIRCUIT(
            REGISTER r0 "offset\\"
        )CIRCUIT");
        },
        std::invalid_argument);

    EXPECT_THROW(
        {
            Circuit(R"CIRCUIT(
            REGISTER r0 ''
        )CIRCUIT");
        },
        std::invalid_argument);

    EXPECT_THROW(
        {
            Circuit(R"CIRCUIT(
            REGISTER "" r0
        )CIRCUIT");
        },
        std::invalid_argument);
}

TEST(circuit, compute_max_condition_depth) {
    ASSERT_EQ(
        Circuit(R"CIRCUIT(
    )CIRCUIT")
            .compute_max_condition_depth(),
        0);

    ASSERT_EQ(
        Circuit(R"CIRCUIT(
        X q0
    )CIRCUIT")
            .compute_max_condition_depth(),
        0);

    ASSERT_EQ(
        Circuit(R"CIRCUIT(
        POP_CONDITION
    )CIRCUIT")
            .compute_max_condition_depth(),
        0);

    ASSERT_EQ(
        Circuit(R"CIRCUIT(
        PUSH_CONDITION if b0
    )CIRCUIT")
            .compute_max_condition_depth(),
        1);

    ASSERT_EQ(
        Circuit(R"CIRCUIT(
        POP_CONDITION
        PUSH_CONDITION if b0
    )CIRCUIT")
            .compute_max_condition_depth(),
        1);

    ASSERT_EQ(
        Circuit(R"CIRCUIT(
        PUSH_CONDITION if b0
        PUSH_CONDITION if b1
        POP_CONDITION
        POP_CONDITION
    )CIRCUIT")
            .compute_max_condition_depth(),
        2);

    ASSERT_EQ(
        Circuit(R"CIRCUIT(
        PUSH_CONDITION if b0
        POP_CONDITION
        PUSH_CONDITION if b1
        POP_CONDITION
    )CIRCUIT")
            .compute_max_condition_depth(),
        1);

    ASSERT_EQ(
        Circuit(R"CIRCUIT(
        PUSH_CONDITION if b0
        PUSH_CONDITION if b1
        PUSH_CONDITION if b2
    )CIRCUIT")
            .compute_max_condition_depth(),
        3);

    ASSERT_EQ(
        Circuit(R"CIRCUIT(
        PUSH_CONDITION if b0
        PUSH_CONDITION if b1
        PUSH_CONDITION if b2
        POP_CONDITION
        POP_CONDITION
        POP_CONDITION
        POP_CONDITION
        POP_CONDITION
        POP_CONDITION
    )CIRCUIT")
            .compute_max_condition_depth(),
        3);

    ASSERT_EQ(
        Circuit(R"CIRCUIT(
        PUSH_CONDITION if b0
        PUSH_CONDITION if b1
        PUSH_CONDITION if b2
        POP_CONDITION
        POP_CONDITION
        POP_CONDITION
        POP_CONDITION
        POP_CONDITION
        POP_CONDITION
        PUSH_CONDITION if b0
        PUSH_CONDITION if b1
    )CIRCUIT")
            .compute_max_condition_depth(),
        3);

    ASSERT_EQ(
        Circuit(R"CIRCUIT(
        PUSH_CONDITION if b0
        PUSH_CONDITION if b1
        PUSH_CONDITION if b2
        POP_CONDITION
        POP_CONDITION
        POP_CONDITION
        POP_CONDITION
        POP_CONDITION
        POP_CONDITION
        PUSH_CONDITION if b0
        PUSH_CONDITION if b1
        PUSH_CONDITION if b2
    )CIRCUIT")
            .compute_max_condition_depth(),
        3);

    ASSERT_EQ(
        Circuit(R"CIRCUIT(
        PUSH_CONDITION if b0
        PUSH_CONDITION if b1
        PUSH_CONDITION if b2
        POP_CONDITION
        POP_CONDITION
        POP_CONDITION
        POP_CONDITION
        POP_CONDITION
        POP_CONDITION
        PUSH_CONDITION if b0
        PUSH_CONDITION if b1
        PUSH_CONDITION if b2
        PUSH_CONDITION if b1
    )CIRCUIT")
            .compute_max_condition_depth(),
        4);
}

TEST(circuit, reaction_depth_bit_invert_preserves_depth) {
    ASSERT_EQ(
        Circuit(R"CIRCUIT(
        HMR q0 b0
        BIT_INVERT b0
        X q1 if b0
    )CIRCUIT")
            .reaction_depth(),
        1);
}

TEST(circuit, parse_z_pow) {
    Circuit c(R"CIRCUIT(
        Z_POW q3 0.125
    )CIRCUIT");
    ASSERT_EQ(c.num_angle_ops, 1);
    ASSERT_EQ(c.angles[0], FixedPrecisionAngle128::from_half_turns_exact_double(0.125));
    ASSERT_EQ(c.num_ops, 1);
    ASSERT_EQ(c.op_types[0], OpType::Z_POW);
    ASSERT_EQ(c.num_q_ops, 1);
    ASSERT_EQ(c.q0[0], 3);
    ASSERT_EQ(c.num_c_ops, 0);
    ASSERT_EQ(c.str(), "Z_POW q3 0.125");

    RaiiTempNamedFile tmp;
    FILE *f = fopen(tmp.path.c_str(), "w");
    c.write_kmx_to(f);
    fclose(f);
    ASSERT_EQ(tmp.read_contents(), "Z_POW q3 0.125\n");
}

TEST(circuit, parse_z_pow_if) {
    Circuit c(R"CIRCUIT(
        Z_POW q3 0.25 if b5
    )CIRCUIT");
    ASSERT_EQ(c.num_angle_ops, 1);
    ASSERT_EQ(c.angles[0], FixedPrecisionAngle128::from_half_turns_exact_double(0.25));
    ASSERT_EQ(c.num_ops, 1);
    ASSERT_EQ(c.op_types[0], OpType::Z_POW_IF);
    ASSERT_EQ(c.num_q_ops, 1);
    ASSERT_EQ(c.q0[0], 3);
    ASSERT_EQ(c.num_c_ops, 1);
    ASSERT_EQ(c.bc[0], 5);
    ASSERT_EQ(c.str(), "Z_POW q3 0.25 if b5");

    RaiiTempNamedFile tmp;
    FILE *f = fopen(tmp.path.c_str(), "w");
    c.write_kmx_to(f);
    fclose(f);
    ASSERT_EQ(tmp.read_contents(), "Z_POW q3 0.25 if b5\n");
}

TEST(circuit, parse_multiple_z_pow) {
    std::string_view expected = "Z_POW q3 0.25 if b5\nZ_POW q5 0.125";
    ASSERT_EQ(Circuit(expected).str(), expected);
}

TEST(op, kmx_str_works_on_every_op) {
    Circuit c = circuit_with_every_operation();
    std::string actual;
    c.iter_ops([&](const Op &op) {
        actual.append(op.kmx_str());
        actual.push_back('\n');
    });
    actual.pop_back();

    c.register_data.clear();
    std::stringstream ss;
    c.write_kmx_to(ss);
    auto expected = ss.str();

    ASSERT_EQ(actual, expected);
}

TEST(circuit, reaction_depth_of_zpow_1_if) {
    Circuit with_z_if(R"CIRCUIT(
        HMR q0 b0
        Z q1 if b0
        HMR q1 b1
        X q2 if b1
    )CIRCUIT");
    Circuit with_z_pow_1_if(R"CIRCUIT(
        HMR q0 b0
        Z_POW q1 1.0 if b0
        HMR q1 b1
        X q2 if b1
    )CIRCUIT");
    EXPECT_EQ(with_z_pow_1_if.reaction_depth(), with_z_if.reaction_depth())
        << "`Z_POW q1 1.0 if b0` and `Z q1 if b0` are physically identical, "
           "but have different reaction_depth()!";
}
