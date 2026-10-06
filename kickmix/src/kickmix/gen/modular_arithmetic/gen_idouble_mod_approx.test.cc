#include "gen_idouble_mod_approx.h"

#include "gtest/gtest.h"

#include "kickmix/sim/fuzzer.h"
#include "kickmix/util/circuit_testing.test.h"
#include "test.test.h"

using namespace kickmix;

TEST(gen_idouble_mod_approx, gen_idouble_mod_approx_below_power_of_2) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    size_t n = 256;
    FixedWidthInt modulus(n);
    modulus ^= -1;

    fuzzer.use_circuit_builder([&](CircuitBuilder &builder, std::mt19937_64 &rng) {
        auto target = builder.append_register(n, "target");
        auto modulus_bottom_bits = builder.append_classical_register_mixed_result(10, "mod_low_bits");
        std::vector<QubitOrBitOrBool> modulus_bits;
        for (size_t k = 0; k < 256; k++) {
            if (k >= 1 && k < 11) {
                modulus_bits.push_back(modulus_bottom_bits[k - 1]);
            } else {
                modulus_bits.push_back(true);
            }
        }
        auto clean = builder.reserve_qubits(std::max(n, static_cast<size_t>(2)) - 2);

        gen_idouble_approx_below_power_of_2(builder, CircuitGenCtx{.clean_workspace = clean}, target, modulus_bits, 30);
    });

    fuzzer.use_input_sampler([&](InputSample &sample, std::mt19937_64 &rng) {
        sample["mod_low_bits"].randomize(rng);
        for (size_t k = 0; k < 10; k++) {
            modulus.bit_ref(k + 1) = sample["mod_low_bits"][k];
        }
        sample["target"].randomize_mod(rng, modulus);
    });

    fuzzer.use_output_sampler([&](OutputSample &sample) {
        for (size_t k = 0; k < 10; k++) {
            modulus.bit_ref(k + 1) = sample["mod_low_bits"][k];
        }
        sample["target"].idouble_mod(modulus);
    });

    fuzzer.fuzz(4, 64);
}

TEST(gen_idouble_mod_approx, gen_ihalve_approx_below_power_of_2_fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    size_t n = 256;
    FixedWidthInt modulus(n);
    modulus ^= -1;

    fuzzer.use_circuit_builder([&](CircuitBuilder &builder, std::mt19937_64 &rng) {
        auto target = builder.append_register(n, "target");
        auto modulus_bottom_bits = builder.append_classical_register_mixed_result(10, "mod_low_bits");
        std::vector<QubitOrBitOrBool> modulus_bits;
        for (size_t k = 0; k < 256; k++) {
            if (k >= 1 && k < 11) {
                modulus_bits.push_back(modulus_bottom_bits[k - 1]);
            } else {
                modulus_bits.push_back(true);
            }
        }
        auto clean = builder.reserve_qubits(std::max(n, static_cast<size_t>(2)) - 2);

        gen_ihalve_approx_below_power_of_2(builder, CircuitGenCtx{.clean_workspace = clean}, target, modulus_bits, 30);
    });

    fuzzer.use_input_sampler([&](InputSample &sample, std::mt19937_64 &rng) {
        sample["mod_low_bits"].randomize(rng);
        for (size_t k = 0; k < 10; k++) {
            modulus.bit_ref(k + 1) = sample["mod_low_bits"][k];
        }
        sample["target"].randomize_mod(rng, modulus);
    });

    fuzzer.use_output_sampler([&](OutputSample &sample) {
        for (size_t k = 0; k < 10; k++) {
            modulus.bit_ref(k + 1) = sample["mod_low_bits"][k];
        }
        sample["target"].ihalve_mod(modulus);
    });

    fuzzer.fuzz(4, 64);
}
