#include "gen_iadd_mod.h"

#include "gtest/gtest.h"

#include "kickmix/sim/fuzzer.h"
#include "kickmix/util/circuit_testing.test.h"
#include "test.test.h"

using namespace kickmix;

TEST(gen_iadd_mod, add_fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t n = 2 + rng() % 10;
        auto target = builder.append_register(n, "target");
        auto offset = builder.append_register(n, "offset");
        auto modulus = builder.append_classical_register_qcarray_result(n, "modulus");
        auto control = builder.append_register(1, "control")[0];
        auto clean = builder.append_register(n + 3, "@clean");
        gen_iadd_mod(builder, CircuitGenCtx{.clean_workspace = clean}, target, offset, modulus, control, INFINITY);
    });

    fuzzer.use_input_sampler([](InputSample &sample, std::mt19937_64 &rng) {
        sample["control"].randomize(rng);
        sample["modulus"].randomize(rng);
        sample["modulus"].bit_ref(sample["modulus"].num_bits - 1) = true;
        sample["target"].randomize_mod(rng, sample["modulus"]);
        sample["offset"].randomize_mod(rng, sample["modulus"]);
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        if (sample["control"]) {
            sample["target"].iadd_mod(sample["offset"], sample["modulus"]);
        }
    });

    fuzzer.fuzz(10, 256);
}

TEST(gen_iadd_mod, sub_fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t n = 2 + rng() % 10;
        auto target = builder.append_register(n, "target");
        auto offset = builder.append_register(n, "offset");
        auto modulus = builder.append_classical_register_qcarray_result(n, "modulus");
        auto control = builder.append_register(1, "control")[0];
        auto clean = builder.append_register(n + 3, "@clean");
        gen_isub_mod(builder, CircuitGenCtx{.clean_workspace = clean}, target, offset, modulus, control, INFINITY);
    });

    fuzzer.use_input_sampler([](InputSample &sample, std::mt19937_64 &rng) {
        sample["control"].randomize(rng);
        sample["modulus"].randomize(rng);
        sample["modulus"].bit_ref(sample["modulus"].num_bits - 1) = true;
        sample["target"].randomize_mod(rng, sample["modulus"]);
        sample["offset"].randomize_mod(rng, sample["modulus"]);
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        if (sample["control"]) {
            sample["target"].isub_mod(sample["offset"], sample["modulus"]);
        }
    });

    fuzzer.fuzz(10, 256);
}
