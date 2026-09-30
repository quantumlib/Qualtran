#include "gtest/gtest.h"

#include "main_util.h"
#include "test.test.h"

using namespace kickmix;

TEST(spin, 64) {
    RaiiTempNamedFile in(R"CIRCUIT(
        #CCX q0 q1 q2
        #CX q0 q1
        #X q0
    )CIRCUIT");

    RaiiTempNamedFile out;
    main_spin(
        10,
        std::vector<const char *>{
            "kickmix",
            "spin",
            "--mode",
            "64",
            "--shots",
            "1024",
            "--in",
            in.path.c_str(),
            "--out",
            out.path.c_str(),
        }
            .data());
    ASSERT_NE(out.read_contents(), "");
}

TEST(spin, 256) {
    RaiiTempNamedFile in(R"CIRCUIT(
        CCX q0 q1 q2
        CX q0 q1
        X q0
    )CIRCUIT");

    RaiiTempNamedFile out;
    main_spin(
        10,
        std::vector<const char *>{
            "kickmix",
            "spin",
            "--mode",
            "256",
            "--shots",
            "1024",
            "--in",
            in.path.c_str(),
            "--out",
            out.path.c_str(),
        }
            .data());
    ASSERT_NE(out.read_contents(), "");
}
