#include "gtest/gtest.h"

#include "main_util.h"
#include "test.test.h"

using namespace kickmix;

TEST(main_diagram, incrementer) {
    RaiiTempNamedFile in(R"CIRCUIT(
        CCX q0 q1 q2
        CX q0 q1
        X q0
    )CIRCUIT");

    RaiiTempNamedFile out;
    main_diagram(
        6,
        std::vector<const char *>{
            "kickmix",
            "diagram",
            "--in",
            in.path.c_str(),
            "--out",
            out.path.c_str(),
        }
            .data());

    ASSERT_EQ("\n" + out.read_contents(), R"DIAGRAM(
q0: -@-@-X-
     | |
q1: -@-X---
     |
q2: -X-----
)DIAGRAM");
}
