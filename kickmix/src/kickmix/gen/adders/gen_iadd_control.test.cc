#include "gen_iadd_control.h"

#include "gtest/gtest.h"

#include "kickmix/sim/fuzzer.h"
#include "kickmix/util/circuit_testing.test.h"
#include "test.test.h"

using namespace kickmix;

TEST(icadd, fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t n = rng() % 10;
        auto target = builder.append_register(n, "target");
        auto offset = builder.append_register(n, "offset");
        auto carry = builder.append_classical_register(1, "carry")[0];
        auto control = builder.append_register(1, "control")[0];
        auto clean = builder.reserve_qubits(n);
        gen_iadd_control(builder, CircuitGenCtx{.clean_workspace = clean}, target, offset, carry, control);
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        if (sample["control"]) {
            sample["target"].iadd_carry(sample["offset"], sample["carry"].non_zero());
        }
    });

    fuzzer.fuzz(20, 256);
}

TEST(icsub, fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t n = rng() % 10;
        auto target = builder.append_register(n, "target");
        auto offset = builder.append_register(n, "offset");
        auto control = builder.append_register(1, "control")[0];
        auto clean = builder.reserve_qubits(n);
        gen_isub_control(builder, CircuitGenCtx{.clean_workspace = clean}, target, offset, control);
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        if (sample["control"]) {
            sample["target"] -= sample["offset"];
        }
    });

    fuzzer.fuzz(20, 256);
}
