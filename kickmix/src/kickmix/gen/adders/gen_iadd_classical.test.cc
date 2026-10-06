#include "gen_iadd_classical.h"

#include "gtest/gtest.h"

#include "kickmix/sim/fuzzer.h"
#include "test.test.h"

using namespace kickmix;

TEST(iadd_classical, fuzz_gen_ixor_carries_from_addition) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t n = rng() % 10;
        size_t m = rng() % (n + 1);
        auto Q_src = builder.append_register(n, "Q_src");
        auto offset = builder.append_classical_register(m, "offset");
        std::vector<QubitId> Q_dst = builder.append_register(n, "Q_dst");
        auto carry = builder.append_register(1, "carry")[0];
        auto control = builder.append_register(1, "control")[0];
        auto dirty = builder.append_register(n == 1 ? 1 : 0, "@dirty");
        CircuitGenCtx ctx{};
        gen_ixor_carries_from_addition(builder, ctx, Q_src, offset, Q_dst, carry, control);
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        if (sample["control"]) {
            FixedWidthInt carries(sample["Q_dst"].num_bits + 1);
            carries ^= sample["Q_src"];
            carries.iadd_carry(sample["offset"], sample["carry"].non_zero());
            carries ^= sample["Q_src"];
            carries ^= sample["offset"];
            carries >>= 1;
            sample["Q_dst"] ^= carries;
        }
    });

    fuzzer.fuzz(10, 64);
}

TEST(iadd_classical, fuzz_gen_iadd_classical_using_2clean_but_with_vented_carries) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t n = rng() % 10;
        int xor_delta = (rng() % 5) - 2;
        auto clean = builder.append_register(2, "@clean");
        auto carry = builder.append_register(1, "carry")[0];
        auto target = builder.append_register(n, "target");
        auto offset = builder.append_classical_register(n, "offset");
        auto control = builder.append_register(1, "control")[0];
        auto Q_carry_xor_target =
            builder.append_register((size_t)std::max(0, (int)n + xor_delta), "Q_carry_xor_target");

        int min_vent_keys = 0;
        min_vent_keys = std::max(min_vent_keys, (int)target.size() - 2);
        min_vent_keys = std::max(min_vent_keys, (int)Q_carry_xor_target.size() - 2);
        auto vent_keys = builder.reserve_bits((size_t)min_vent_keys);

        auto secret_uncontrolled_carry = builder.append_classical_register(n + 1, "__secret_uncontrolled_carry");
        CircuitGenCtx ctx{.clean_workspace = clean};
        gen_iadd_classical_using_2clean_but_with_vented_carries(
            builder, ctx, target, offset, carry, Q_carry_xor_target, vent_keys, control);

        // Clear expected phase via the vent bits.
        for (size_t k = 0; k < secret_uncontrolled_carry.size() && k < vent_keys.size(); k++) {
            builder.cz(vent_keys[k], secret_uncontrolled_carry[k]);
        }
    });

    fuzzer.use_input_sampler([](InputSample &sample, std::mt19937_64 &rng) {
        sample["carry"].randomize(rng);
        sample["target"].randomize(rng);
        sample["offset"].randomize(rng);
        sample["control"].randomize(rng);
        sample["Q_carry_xor_target"].randomize(rng);

        sample["__secret_uncontrolled_carry"].clear_to_zero();
        if (sample["control"]) {
            sample["__secret_uncontrolled_carry"] ^= sample["target"];
            sample["__secret_uncontrolled_carry"].iadd_carry(sample["offset"], sample["carry"].non_zero());
            sample["__secret_uncontrolled_carry"] ^= sample["target"];
            sample["__secret_uncontrolled_carry"] ^= sample["offset"];
        }
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        if (sample["control"]) {
            sample["Q_carry_xor_target"] ^= sample["__secret_uncontrolled_carry"];
            sample["target"].iadd_carry(sample["offset"], sample["carry"].non_zero());
        }
    });

    fuzzer.fuzz(100, 256);
}

TEST(iadd_classical, fuzz_gen_iadd_classical) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t n = rng() % 10;
        size_t n_clean = 2 + rng() % 10;
        auto target = builder.append_register(n, "target");
        auto offset = builder.append_classical_register(n, "offset");
        auto control = builder.append_register(1, "control")[0];
        auto carry = builder.append_classical_register_qcarray_result(1, "carry")[0];
        auto clean = builder.append_register(n_clean, "@clean");
        auto dirty = builder.append_register(n, "@dirty");
        CircuitGenCtx ctx{.clean_workspace = clean};
        ctx = ctx.with_more_dirty_qubits(dirty);
        gen_iadd_classical(builder, ctx, target, offset, carry, control, INFINITY);
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        if (sample["control"]) {
            sample["target"].iadd_carry(sample["offset"], sample["carry"].non_zero());
        }
    });

    fuzzer.fuzz(100, 256);
}

TEST(iadd_classical, fuzz_gen_iadd_classical_simple) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t n = rng() % 16;
        size_t n_clean = n;
        auto target = builder.append_register(n, "target");
        auto offset = builder.append_classical_register_qcarray_result(n, "offset");
        auto control = builder.append_register(1, "control")[0];
        auto carry = builder.append_classical_register_mixed_result(1, "carry")[0];
        auto clean = builder.append_register(n_clean, "@clean");
        auto dirty = builder.append_register(n - n_clean, "@dirty");
        CircuitGenCtx ctx{.clean_workspace = clean};
        ctx = ctx.with_more_dirty_qubits(dirty);
        gen_iadd_classical_simple(builder, ctx, target, offset, carry, control);
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        if (sample["control"]) {
            sample["target"] += sample["offset"];
            sample["target"] += sample["carry"];
        }
    });

    fuzzer.fuzz(32, 64);
}

TEST(isub_classical, fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t n = rng() % 8;
        auto target = builder.append_register(n, "target");
        auto offset = builder.append_classical_register_qcarray_result(n, "offset");
        auto control = builder.append_register(1, "control")[0];
        auto borrow = builder.append_classical_register_mixed_result(1, "borrow")[0];
        auto clean = builder.append_register(n, "@clean");
        gen_isub_classical(builder, CircuitGenCtx{.clean_workspace = clean}, target, offset, borrow, control, INFINITY);
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        if (sample["control"]) {
            sample["target"] -= sample["offset"];
            sample["target"] -= sample["borrow"];
        }
    });

    fuzzer.fuzz(16, 64);
}
