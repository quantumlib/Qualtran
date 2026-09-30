#include "gen_gf_inverse.h"

#include "gtest/gtest.h"

#include "kickmix/sim/fuzzer.h"
#include "test.test.h"

using namespace kickmix;

/// Writes the low bits of a polynomial into a fuzzer register, leaving its width alone.
static void write_poly(FixedWidthInt &target, const GF2Poly &value) {
    for (size_t k = 0; k < target.num_bits; k++) {
        target.bit_ref(k) = value.bit(k);
    }
}

/// The inverse of a field element, with zero mapping to itself so the map stays reversible.
static GF2Poly reference_inverse(const GF2Field &field, const GF2Poly &value) {
    if (value.is_zero()) {
        return value;
    }
    return field.invert(value);
}

TEST(gen_gf_inverse, fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());
    auto field = std::make_shared<std::optional<GF2Field>>();

    fuzzer.use_circuit_builder([&](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t m = 1 + rng() % 8;
        *field = GF2Field(m);
        auto input = builder.append_register(m, "input");
        auto target = builder.append_register(m, "target");
        auto clean = builder.append_register(gf_inverse_workspace_size(**field), "@clean");
        gen_gf_inverse(builder, CircuitGenCtx{.clean_workspace = clean}, **field, target, input);
    });

    fuzzer.use_input_and_output_sampler([&](InputSample &ins, OutputSample &outs, std::mt19937_64 &rng) {
        ins["input"].randomize(rng);
        // The target has to start zeroed; the inverse is written into it, not added into it.
        write_poly(ins["target"], GF2Poly{});
        outs["input"] = ins["input"];
        write_poly(outs["target"], reference_inverse(**field, GF2Poly::from_fixed_width_int(ins["input"])));
    });

    fuzzer.fuzz(60, 64);
}

TEST(gen_gf_inverse, round_trip_with_clean_workspace_fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([&](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t m = 1 + rng() % 8;
        GF2Field field(m);
        auto input = builder.append_register(m, "input");
        auto clean = builder.append_register(m + gf_inverse_workspace_size(field), "@clean");
        std::span<const QubitId> clean_span(clean);
        auto target = clean_span.subspan(0, m);
        CircuitGenCtx ctx{.clean_workspace = clean_span.subspan(m)};
        gen_gf_inverse(builder, ctx, field, target, input);
        gen_gf_uninverse(builder, ctx, field, target, input);
    });

    fuzzer.fuzz(50, 64);
}

TEST(gen_gf_inverse, round_trip_with_explicit_chain_fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([&](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t m = 1 + rng() % 8;
        GF2Field field(m);
        auto input = builder.append_register(m, "input");
        auto clean = builder.append_register(m + gf_inverse_chain_size(field), "@clean");
        std::span<const QubitId> clean_span(clean);
        auto target = clean_span.subspan(0, m);
        auto chain = clean_span.subspan(m);
        gen_gf_inverse(builder, CircuitGenCtx{}, field, target, input, chain);
        gen_gf_uninverse(builder, CircuitGenCtx{}, field, target, input, chain);
    });

    fuzzer.fuzz(50, 64);
}

TEST(gen_gf_inverse, explicit_chain_saves_toffolis_on_uncompute) {
    GF2Field field(64);
    size_t self_cleaning_round_trip_toffolis;
    size_t explicit_chain_round_trip_toffolis;
    size_t uninverse_with_chain_toffolis;
    {
        CircuitBuilder builder;
        auto input = builder.append_register(64);
        auto target = builder.append_register(64);
        auto clean = builder.reserve_qubits(gf_inverse_workspace_size(field));
        CircuitGenCtx ctx{.clean_workspace = clean};
        gen_gf_inverse(builder, ctx, field, target, input);
        gen_gf_uninverse(builder, ctx, field, target, input);
        self_cleaning_round_trip_toffolis = builder.finish_circuit().max_magic();
    }
    {
        CircuitBuilder builder;
        auto input = builder.append_register(64);
        auto target = builder.append_register(64);
        auto chain = builder.reserve_qubits(gf_inverse_chain_size(field));
        gen_gf_inverse(builder, CircuitGenCtx{}, field, target, input, chain);
        gen_gf_uninverse(builder, CircuitGenCtx{}, field, target, input, chain);
        explicit_chain_round_trip_toffolis = builder.finish_circuit().max_magic();
    }
    {
        CircuitBuilder builder;
        auto input = builder.append_register(64);
        auto target = builder.append_register(64);
        auto chain = builder.reserve_qubits(gf_inverse_chain_size(field));
        gen_gf_uninverse(builder, CircuitGenCtx{}, field, target, input, chain);
        uninverse_with_chain_toffolis = builder.finish_circuit().max_magic();
    }
    ASSERT_EQ(uninverse_with_chain_toffolis, 0);
    ASSERT_EQ(explicit_chain_round_trip_toffolis, 7290);
    ASSERT_EQ(self_cleaning_round_trip_toffolis, 14580);
    ASSERT_LT(gf_inverse_chain_size(field), gf_inverse_workspace_size(field));
}

TEST(gen_gf_inverse, is_an_involution_fuzz) {
    // Inverting twice must return the original value, which is a property the reference model used
    // by the fuzz test above cannot accidentally satisfy.
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t m = 2 + rng() % 7;
        GF2Field field(m);
        auto input = builder.append_register(m, "input");
        auto twice = builder.append_register(m, "twice");
        auto clean = builder.append_register(m + gf_inverse_workspace_size(field), "@clean");
        std::span<const QubitId> clean_span(clean);
        auto once = clean_span.subspan(0, m);
        CircuitGenCtx ctx{.clean_workspace = clean_span.subspan(m)};
        gen_gf_inverse(builder, ctx, field, once, input);
        gen_gf_inverse(builder, ctx, field, twice, once);
        gen_gf_uninverse(builder, ctx, field, once, input);
    });

    fuzzer.use_input_sampler([](InputSample &sample, std::mt19937_64 &rng) {
        sample["input"].randomize(rng);
        write_poly(sample["twice"], GF2Poly{});
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        write_poly(sample["twice"], GF2Poly::from_fixed_width_int(sample["input"]));
    });

    fuzzer.fuzz(40, 64);
}

TEST(gen_gf_inverse, workspace_size_matches_what_is_consumed) {
    for (size_t m : std::vector<size_t>{1, 2, 3, 4, 5, 8, 16, 17, 64, 128}) {
        GF2Field field(m);
        size_t clean_size = gf_inverse_workspace_size(field);
        size_t chain_size = gf_inverse_chain_size(field);
        ASSERT_EQ(clean_size % m, 0) << m;
        ASSERT_EQ(chain_size % m, 0) << m;
        ASSERT_LE(chain_size, clean_size) << m;
        // Every chain register is one field element wide, and there are O(log m) of them.
        ASSERT_LE(clean_size, 2 * m * (1 + std::bit_width(m))) << m;

        // Exactly the advertised amount must be enough; taking more would throw.
        CircuitBuilder builder;
        auto input = builder.append_register(m);
        auto target = builder.append_register(m);
        auto clean = builder.reserve_qubits(clean_size);
        gen_gf_inverse(builder, CircuitGenCtx{.clean_workspace = clean}, field, target, input);
    }
}

TEST(gen_gf_inverse, rejects_missing_workspace) {
    GF2Field field(8);
    CircuitBuilder builder;
    auto input = builder.append_register(8);
    auto target = builder.append_register(8);
    ASSERT_THROW({ gen_gf_inverse(builder, CircuitGenCtx{}, field, target, input); }, std::invalid_argument);
    ASSERT_THROW({ gen_gf_uninverse(builder, CircuitGenCtx{}, field, target, input); }, std::invalid_argument);
}

TEST(gen_gf_inverse, rejects_bad_register_sizes) {
    GF2Field field(8);
    CircuitBuilder builder;
    auto good = builder.append_register(8);
    auto bad = builder.append_register(7);
    auto clean = builder.reserve_qubits(gf_inverse_workspace_size(field));
    CircuitGenCtx ctx{.clean_workspace = clean};
    ASSERT_THROW({ gen_gf_inverse(builder, ctx, field, bad, good); }, std::invalid_argument);
    ASSERT_THROW({ gen_gf_inverse(builder, ctx, field, good, bad); }, std::invalid_argument);
    ASSERT_THROW({ gen_gf_inverse(builder, ctx, field, good, good, bad); }, std::invalid_argument);
    ASSERT_THROW({ gen_gf_uninverse(builder, ctx, field, good, good, bad); }, std::invalid_argument);
}

/// Asserts that every qubit in the circuit follows a strict reset -> use -> measure-out lifecycle.
///
/// Reusing a workspace qubit after measuring it out, without an intervening reset, still simulates
/// correctly (an HMR leaves the qubit holding zero), so the value based fuzz tests are blind to it.
/// It does corrupt qubit allocation accounting and circuit rendering, so it is checked structurally.
static void expect_balanced_qubit_lifecycle(const Circuit &circuit, std::string_view context) {
    std::vector<bool> active(circuit.num_qubits, false);
    auto require_active = [&](QubitOrFalse q, const char *role) {
        if (q.is_qubit()) {
            ASSERT_TRUE(active[q.untagged_id()]) << context << ": op used measured-out " << role;
        }
    };
    circuit.iter_ops([&](const Op &op) {
        if (op.kind == OpType::R) {
            ASSERT_FALSE(active[op.q_target.untagged_id()]) << context << ": qubit reset while still active";
            active[op.q_target.untagged_id()] = true;
        } else if (op.kind == OpType::HMR) {
            ASSERT_TRUE(active[op.q_target.untagged_id()]) << context << ": qubit measured out twice";
            active[op.q_target.untagged_id()] = false;
        } else {
            require_active(op.q_target, "target");
            require_active(op.q_control1, "control1");
            require_active(op.q_control2, "control2");
        }
    });
    for (size_t q = 0; q < circuit.num_qubits; q++) {
        ASSERT_FALSE(active[q]) << context << ": qubit " << q << " was never freed";
    }
}

TEST(gen_gf_inverse, qubit_lifecycle_is_balanced_with_clean_workspace) {
    for (size_t m : std::vector<size_t>{2, 3, 4, 5, 6, 7, 8, 9, 16, 17}) {
        GF2Field field(m);
        CircuitBuilder builder;
        auto input = builder.append_register(m, "input");
        auto target = builder.append_register(m, "target");
        auto clean = builder.append_register(gf_inverse_workspace_size(field), "@clean");
        builder.broadcast_reset(input);
        builder.broadcast_reset(target);
        CircuitGenCtx ctx{.clean_workspace = clean};
        gen_gf_inverse(builder, ctx, field, target, input);
        gen_gf_uninverse(builder, ctx, field, target, input);
        builder.broadcast_del_zero(input);
        expect_balanced_qubit_lifecycle(builder.finish_circuit(), "clean workspace m=" + std::to_string(m));
    }
}

TEST(gen_gf_inverse, qubit_lifecycle_is_balanced_with_explicit_chain) {
    // The explicit chain overloads unwind the fold steps with gen_gf_unmul, which measures out the
    // output register before the doubling cleanup wants to borrow it back as scratch.
    for (size_t m : std::vector<size_t>{2, 3, 4, 5, 6, 7, 8, 9, 16, 17}) {
        GF2Field field(m);
        CircuitBuilder builder;
        auto input = builder.append_register(m, "input");
        auto target = builder.append_register(m, "target");
        auto chain = builder.append_register(gf_inverse_chain_size(field), "@clean");
        builder.broadcast_reset(input);
        builder.broadcast_reset(target);
        builder.broadcast_reset(chain);
        gen_gf_inverse(builder, CircuitGenCtx{}, field, target, input, chain);
        gen_gf_uninverse(builder, CircuitGenCtx{}, field, target, input, chain);
        builder.broadcast_del_zero(input);
        expect_balanced_qubit_lifecycle(builder.finish_circuit(), "explicit chain m=" + std::to_string(m));
    }
}

TEST(gen_gf_inverse, gf4_inversion_never_swaps_the_input) {
    // For m = 2 the addition chain is empty (k1 == 0), so the inverse is a copy plus a Frobenius.
    // An unguarded broadcast_swap(f[k1], f[k]) would degenerate into swapping the input register.
    GF2Field field(2);
    for (bool explicit_chain : std::vector<bool>{false, true}) {
        CircuitBuilder builder;
        auto input = builder.append_register(2, "input");
        auto target = builder.append_register(2, "target");
        if (explicit_chain) {
            ASSERT_EQ(gf_inverse_chain_size(field), 0);
            gen_gf_inverse(builder, CircuitGenCtx{}, field, target, input, {});
        } else {
            gen_gf_inverse(builder, CircuitGenCtx{}, field, target, input);
        }
        Circuit circuit = builder.finish_circuit();
        circuit.iter_ops([&](const Op &op) {
            if (op.kind == OpType::SWAP) {
                for (QubitId q : input) {
                    ASSERT_NE(op.q_target.untagged_id(), q.untagged_id()) << "swap touched the input";
                    ASSERT_NE(op.q_control1.untagged_id(), q.untagged_id()) << "swap touched the input";
                }
            }
        });
    }
}
