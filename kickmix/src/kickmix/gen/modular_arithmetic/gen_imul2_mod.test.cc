#include "gen_imul2_mod.h"

#include "gtest/gtest.h"

#include "kickmix/sim/fuzzer.h"
#include "test.test.h"

using namespace kickmix;

TEST(imul2_mod_approx, fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());
    size_t n = 32;
    FixedWidthInt modulus(n);
    modulus ^= -1;
    modulus ^= 1 << 25;
    array_z modulus_bits = array_z::copy_of(modulus);

    fuzzer.use_circuit_builder([&](CircuitBuilder &builder, std::mt19937_64 &rng) {
        auto target = builder.append_register(n, "target");
        auto clean = builder.append_register(4, "@clean");
        auto dirty = builder.append_register(n, "@dirty");
        double btol = 32;
        gen_imul2_mod(
            builder,
            CircuitGenCtx{.clean_workspace = clean}.with_more_dirty_qubits(dirty),
            target,
            modulus_bits,
            true,
            btol);
    });

    fuzzer.use_input_sampler([&](InputSample &sample, std::mt19937_64 &rng) {
        sample["target"].randomize_mod(rng, modulus);
    });

    fuzzer.use_output_sampler([&](OutputSample &sample) {
        sample["target"].idouble_mod(modulus);
    });

    fuzzer.fuzz(1, 256);
}

TEST(imul2_inv_mod, fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([&](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t n = 1 + rng() % 20;
        auto target = builder.append_register(n, "target");
        auto modulus_bits = builder.append_classical_register_qcarray_result(n, "mod");
        modulus_bits.front() = true;
        modulus_bits.back() = true;
        modulus_bits.common_type = QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_ZLIKE;
        auto control = builder.append_register(1, "control")[0];
        auto clean = builder.append_register(std::max(n, static_cast<size_t>(2)) - 1, "@clean");
        gen_imul2_inv_mod_with_subcmp_merged_by_inlining(
            builder, CircuitGenCtx{.clean_workspace = clean}, target, modulus_bits, control);
    });

    fuzzer.use_input_sampler([&](InputSample &sample, std::mt19937_64 &rng) {
        sample["mod"].randomize(rng);
        sample["mod"].back_ref() = true;
        sample["mod"].front_ref() = true;
        sample["target"].randomize_mod(rng, sample["mod"]);
        sample["control"].randomize(rng);
    });

    fuzzer.use_output_sampler([&](OutputSample &sample) {
        if (sample["control"]) {
            sample["target"].ihalve_mod(sample["mod"]);
        }
    });

    fuzzer.fuzz(20, 256);
}

TEST(imul2_inv_mod, approx_fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());
    auto modulus = FixedWidthInt("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    size_t n = modulus.num_bits;
    array_z modulus_bits = array_z::copy_of(modulus);

    fuzzer.use_circuit_builder([&](CircuitBuilder &builder, std::mt19937_64 &rng) {
        auto target = builder.append_register(n, "target");
        auto control = builder.append_register(1, "control")[0];
        auto clean = builder.append_register(n + 10, "@clean");
        gen_imul2_inv_mod(builder, CircuitGenCtx{.clean_workspace = clean}, target, modulus_bits, control, 20);
    });

    fuzzer.use_input_sampler([&](InputSample &sample, std::mt19937_64 &rng) {
        sample["target"].randomize_mod(rng, modulus);
        sample["control"].randomize(rng);
    });

    fuzzer.use_output_sampler([&](OutputSample &sample) {
        if (sample["control"]) {
            sample["target"].ihalve_mod(modulus);
        }
    });

    fuzzer.fuzz(1, 256);
}

TEST(imul2_mod, fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        bool minimize = rng() % 2;
        size_t n = rng() % 20 + 1;

        auto target = builder.append_register(n, "target");
        auto modulus_bits = builder.append_classical_register_qcarray_result(n, "modulus");
        modulus_bits.front() = true;
        modulus_bits.back() = true;
        modulus_bits.common_type = QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_ZLIKE;
        auto clean = builder.append_register(n + 3, "@clean");
        auto control = builder.append_register(1, "control")[0];
        auto dirty = builder.append_register(n + 3, "@dirty");
        auto ctx =
            CircuitGenCtx{.clean_workspace = clean, .minimize_qubits = minimize != 0}.with_more_dirty_qubits(dirty);
        gen_imul2_mod(builder, ctx, target, modulus_bits, control, INFINITY);
    });

    fuzzer.use_input_sampler([](InputSample &sample, std::mt19937_64 &rng) {
        sample["modulus"].randomize(rng);
        sample["modulus"].bit_ref(0) = true;
        sample["modulus"].bit_ref(sample["modulus"].num_bits - 1) = true;
        sample["target"].randomize_mod(rng, sample["modulus"]);
        sample["control"].randomize(rng);
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        if (sample["control"]) {
            sample["target"].idouble_mod(sample["modulus"]);
        }
    });

    fuzzer.fuzz(10, 128);
}
