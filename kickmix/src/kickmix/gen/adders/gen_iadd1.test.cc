#include "gen_iadd1.h"

#include "gtest/gtest.h"

#include "kickmix/sim/fuzzer.h"
#include "kickmix/util/circuit_testing.test.h"
#include "test.test.h"

using namespace kickmix;

TEST(gen_iadd1, fuzz_no_control) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t n = rng() % 10;
        auto target = builder.append_register(n, "target");
        auto clean = builder.append_register(n, "@clean");
        gen_iadd1(builder, CircuitGenCtx{.clean_workspace = clean}, target);
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        sample["target"].increment();
    });

    fuzzer.fuzz(50, 256);
}

TEST(gen_iadd1, fuzz_control) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t n = rng() % 10;
        auto target = builder.append_register(n, "target");
        auto clean = builder.append_register(n, "@clean");
        auto control = builder.append_register(1, "control")[0];
        gen_iadd1(builder, CircuitGenCtx{.clean_workspace = clean}, target, control);
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        sample["target"] += sample["control"];
    });

    fuzzer.fuzz(50, 256);
}
