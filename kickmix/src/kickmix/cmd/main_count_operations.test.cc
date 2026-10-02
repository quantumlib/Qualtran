#include "gtest/gtest.h"

#include "kickmix/util/circuit_testing.test.h"
#include "main_util.h"
#include "test.test.h"

using namespace kickmix;

TEST(main_count_operations, every_operation) {
    RaiiTempNamedFile in(circuit_with_every_operation().str());

    RaiiTempNamedFile out;
    main_count_operations(
        7,
        std::vector<const char *>{
            "kickmix",
            "count_operations",
            "--groups",
            "--in",
            in.path.c_str(),
            "--out",
            out.path.c_str(),
        }
            .data());

    ASSERT_NE(out.read_contents(), "");
}
