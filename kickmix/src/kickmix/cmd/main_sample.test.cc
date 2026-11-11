#include "gtest/gtest.h"

#include "main_util.h"
#include "test.test.h"

using namespace kickmix;

TEST(main_sample, named_and_unnamed_registers) {
    RaiiTempNamedFile in(R"CIRCUIT(
        REGISTER r0 "my_reg"
        APPEND_TO_REGISTER q0 r0
        APPEND_TO_REGISTER q1 r0
        APPEND_TO_REGISTER b0 r1
        X q0
        BIT_INVERT b0
    )CIRCUIT");

    RaiiTempNamedFile out;
    ASSERT_EQ(
        0,
        main_sample(
            8,
            std::vector<const char *>{
                "kickmix",
                "sample",
                "--in",
                in.path.c_str(),
                "--out",
                out.path.c_str(),
                "--shots",
                "2",
            }
                .data()));

    ASSERT_EQ("\n" + out.read_contents(), R"OUTPUT(
(before)my_reg,my_reg,(before)r1,r1,global_phase
     00,     01,      0,      1,           1
     00,     01,      0,      1,           1
)OUTPUT");
}
