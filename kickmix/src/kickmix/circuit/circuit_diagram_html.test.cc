#include <fstream>

#include "gtest/gtest.h"

#include "kickmix/circuit/circuit.h"
#include "kickmix/util/circuit_testing.test.h"

using namespace kickmix;

TEST(circuit, html_diagram) {
    Circuit c(R"CIRCUIT(
         APPEND_TO_REGISTER q0 r0
         APPEND_TO_REGISTER q1 r0
         APPEND_TO_REGISTER q2 r0
         APPEND_TO_REGISTER q3 r0
         APPEND_TO_REGISTER q4 r0
         APPEND_TO_REGISTER q5 r0
         APPEND_TO_REGISTER q6 r0
         APPEND_TO_REGISTER q7 r0

         APPEND_TO_REGISTER q12 r1
         APPEND_TO_REGISTER q13 r1
         APPEND_TO_REGISTER q14 r1
         APPEND_TO_REGISTER q15 r1
         APPEND_TO_REGISTER q16 r1

         CCX q0 q1 q12
         CCX q2 q12 q13
         CCX q3 q13 q14
         CCX q4 q14 q15
         CCX q5 q15 q16

         CCX q16 q6 q7

         CX q16 q6
         HMR q16 b0
         CZ q5 q15 if b0

         CX q15 q5
         HMR q15 b0
         CZ q4 q14 if b0

         CX q14 q4
         HMR q14 b0
         CZ q3 q13 if b0

         CX q13 q3
         HMR q13 b0
         CZ q2 q12 if b0

         CX q12 q2
         HMR q12 b0
         CZ q0 q1 if b0

         CX q0 q1
         X q0
    )CIRCUIT");
    auto s = c.html_diagram();
    ASSERT_TRUE(!s.empty());
}

TEST(circuit, html_diagram_fused_cx) {
    Circuit c(R"CIRCUIT(
        CX q5 q3 if b0
        CX q5 q4
        CX q5 q7
        CX q5 q8
        CX q5 q7
    )CIRCUIT");
    auto s = c.html_diagram();
    ASSERT_TRUE(!s.empty());
}

TEST(circuit, html_diagram_works_on_all_gates) {
    auto c = circuit_with_every_operation();
    auto s = c.html_diagram();
    ASSERT_TRUE(!s.empty());
}

TEST(circuit, html_diagram_info_tags) {
    Circuit c(R"CIRCUIT(
        INFO_TAG "ignored_tag"
        INFO_TAG "@exit:unmatched"
        INFO_TAG "@enter:outer_sub"
        X q0
        INFO_TAG "@enter:inner_sub"
        CX q0 q1
        INFO_TAG "@exit:inner_sub"
        CCX q0 q1 q2
        INFO_TAG "@exit:outer_sub"
        INFO_TAG "@enter:unclosed_sub"
        Z q0
    )CIRCUIT");
    auto s = c.html_diagram();
    ASSERT_NE(s.find("outer_sub"), std::string::npos);
    ASSERT_NE(s.find("inner_sub"), std::string::npos);
    ASSERT_NE(s.find("unclosed_sub"), std::string::npos);
    ASSERT_EQ(s.find("ignored_tag"), std::string::npos);
    ASSERT_EQ(s.find("unmatched"), std::string::npos);
}

TEST(circuit, html_diagram_register_names) {
    Circuit c(R"CIRCUIT(
        REGISTER r0 "my_qreg"
        APPEND_TO_REGISTER q0 r0
        APPEND_TO_REGISTER q1 r0
        REGISTER r1 "my_breg"
        APPEND_TO_REGISTER b0 r1
        APPEND_TO_REGISTER b1 r1
        X q0
        BIT_INVERT b0
    )CIRCUIT");
    auto s = c.html_diagram();
    ASSERT_NE(s.find("my_qreg[0]"), std::string::npos);
    ASSERT_NE(s.find("my_breg[0]=b0"), std::string::npos);
    ASSERT_EQ(s.find("]==b"), std::string::npos);
}
