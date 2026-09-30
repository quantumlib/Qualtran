#include "gen_ineg_mod.h"

#include "gtest/gtest.h"

#include "kickmix/sim/fuzzer.h"
#include "kickmix/util/circuit_testing.test.h"
#include "test.test.h"

using namespace kickmix;

TEST(ineg_mod, fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t n = 1 + rng() % 20;
        auto target = builder.append_register(n, "target");
        auto modulus_bits = builder.append_classical_register_qcarray_result(n, "mod");
        modulus_bits.front() = true;
        modulus_bits.back() = true;
        modulus_bits.common_type = QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_ZLIKE;
        auto clean = builder.append_register(n + 1, "@clean");
        gen_ineg_mod(builder, CircuitGenCtx{.clean_workspace = clean}, target, modulus_bits);
    });

    fuzzer.use_input_sampler([](InputSample &sample, std::mt19937_64 &rng) {
        sample["mod"].randomize(rng);
        sample["mod"].front_ref() = true;
        sample["mod"].back_ref() = true;
        sample["target"].randomize_mod(rng, sample["mod"]);
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        sample["target"].negate_mod(sample["mod"]);
    });

    fuzzer.fuzz(20, 64);
}
