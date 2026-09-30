#include "gtest/gtest.h"

#include "kickmix/util/circuit_testing.test.h"
#include "main_util.h"
#include "test.test.h"

using namespace kickmix;

TEST(main_count_operations_sampled, every_operation) {
    RaiiTempNamedFile in(circuit_with_every_operation().str());

    RaiiTempNamedFile out;
    main_count_operations_sampled(
        8,
        std::vector<const char *>{
            "kickmix",
            "count_operations_sampled",
            "--in",
            in.path.c_str(),
            "--out",
            out.path.c_str(),
            "--shots",
            "64",
        }
            .data());

    ASSERT_NE(out.read_contents(), "");
}
