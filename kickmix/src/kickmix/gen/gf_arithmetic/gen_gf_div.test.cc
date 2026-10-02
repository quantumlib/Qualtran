#include "gen_gf_div.h"

#include "gtest/gtest.h"

#include "kickmix/gen/gf_arithmetic/gen_gf_inverse.h"
#include "kickmix/gen/gf_arithmetic/gen_gf_mul.h"
#include "kickmix/sim/fuzzer.h"
#include "test.test.h"

using namespace kickmix;

/// Writes the low bits of a polynomial into a fuzzer register, leaving its width alone.
static void write_poly(FixedWidthInt &target, const GF2Poly &value) {
    for (size_t k = 0; k < target.num_bits; k++) {
        target.bit_ref(k) = value.bit(k);
    }
}

TEST(gen_gf_div, fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());
    auto field = std::make_shared<std::optional<GF2Field>>();

    fuzzer.use_circuit_builder([&](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t m = 1 + rng() % 8;
        *field = GF2Field(m);
        auto lhs = builder.append_register(m, "lhs");
        auto rhs = builder.append_register(m, "rhs");
        auto target = builder.append_register(m, "target");
        auto clean = builder.append_register(gf_div_workspace_size(**field), "@clean");
        gen_gf_div(builder, CircuitGenCtx{.clean_workspace = clean}, **field, target, lhs, rhs);
    });

    fuzzer.use_output_sampler([&](OutputSample &sample) {
        GF2Poly lhs = GF2Poly::from_fixed_width_int(sample["lhs"]);
        GF2Poly rhs = GF2Poly::from_fixed_width_int(sample["rhs"]);
        FixedWidthInt &target = sample["target"];
        GF2Poly v = GF2Poly::from_fixed_width_int(target);
        // div returns zero for a zero divisor, which keeps the whole map reversible.
        write_poly(target, v ^ (*field)->div(lhs, rhs));
    });

    fuzzer.fuzz(50, 64);
}

TEST(gen_gf_div, undiv_undoes_div_fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t m = 1 + rng() % 8;
        GF2Field field(m);
        auto lhs = builder.append_register(m, "lhs");
        auto rhs = builder.append_register(m, "rhs");
        auto clean = builder.append_register(m + gf_div_workspace_size(field), "@clean");
        std::span<const QubitId> clean_span(clean);
        auto target = clean_span.subspan(0, m);
        CircuitGenCtx ctx{.clean_workspace = clean_span.subspan(m)};
        gen_gf_div(builder, ctx, field, target, lhs, rhs);
        gen_gf_undiv(builder, ctx, field, target, lhs, rhs);
    });

    fuzzer.fuzz(40, 64);
}

TEST(gen_gf_div, multiplying_the_quotient_back_recovers_the_dividend_fuzz) {
    // (lhs / rhs) * rhs == lhs whenever rhs is invertible, which is a check independent of the
    // classical reference implementation of division.
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t m = 2 + rng() % 6;
        GF2Field field(m);
        auto lhs = builder.append_register(m, "lhs");
        auto rhs = builder.append_register(m, "rhs");
        auto recovered = builder.append_register(m, "recovered");
        auto clean = builder.append_register(m + gf_div_workspace_size(field), "@clean");
        std::span<const QubitId> clean_span(clean);
        auto quotient = clean_span.subspan(0, m);
        CircuitGenCtx ctx{.clean_workspace = clean_span.subspan(m)};
        gen_gf_div(builder, ctx, field, quotient, lhs, rhs);
        gen_gf_mul(builder, ctx, field, recovered, quotient, rhs);
        gen_gf_undiv(builder, ctx, field, quotient, lhs, rhs);
    });

    fuzzer.use_input_sampler([](InputSample &sample, std::mt19937_64 &rng) {
        sample["lhs"].randomize(rng);
        sample["rhs"].randomize(rng);
        // A zero divisor makes the quotient zero, so nothing would be recovered.
        sample["rhs"].front_ref() = true;
        write_poly(sample["recovered"], GF2Poly{});
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        write_poly(sample["recovered"], GF2Poly::from_fixed_width_int(sample["lhs"]));
    });

    fuzzer.fuzz(40, 64);
}

TEST(gen_gf_div, workspace_and_cost_accounting) {
    GF2Field field(32);
    ASSERT_EQ(gf_div_workspace_size(field), 32 + gf_inverse_chain_size(field));

    // Dividing must cost about one clean inversion plus one multiplication, not two inversions.
    CircuitBuilder div_builder;
    auto lhs = div_builder.append_register(32);
    auto rhs = div_builder.append_register(32);
    auto target = div_builder.append_register(32);
    auto clean = div_builder.reserve_qubits(gf_div_workspace_size(field));
    gen_gf_div(div_builder, CircuitGenCtx{.clean_workspace = clean}, field, target, lhs, rhs);
    size_t div_ops = div_builder.finish_circuit().max_magic();

    CircuitBuilder inv_builder;
    auto in2 = inv_builder.append_register(32);
    auto out2 = inv_builder.append_register(32);
    auto clean2 = inv_builder.reserve_qubits(gf_inverse_workspace_size(field));
    gen_gf_inverse(inv_builder, CircuitGenCtx{.clean_workspace = clean2}, field, out2, in2);
    size_t inv_ops = inv_builder.finish_circuit().max_magic();

    ASSERT_LT(div_ops, 2 * inv_ops) << div_ops << " vs " << inv_ops;
}

TEST(gen_gf_div, rejects_bad_arguments) {
    GF2Field field(8);
    CircuitBuilder builder;
    auto good = builder.append_register(8);
    auto bad = builder.append_register(7);
    auto clean = builder.reserve_qubits(gf_div_workspace_size(field));
    CircuitGenCtx ctx{.clean_workspace = clean};
    ASSERT_THROW({ gen_gf_div(builder, ctx, field, bad, good, good); }, std::invalid_argument);
    ASSERT_THROW({ gen_gf_div(builder, ctx, field, good, bad, good); }, std::invalid_argument);
    ASSERT_THROW({ gen_gf_div(builder, ctx, field, good, good, bad); }, std::invalid_argument);
    ASSERT_THROW({ gen_gf_div(builder, CircuitGenCtx{}, field, good, good, good); }, std::invalid_argument);
    ASSERT_THROW({ gen_gf_undiv(builder, CircuitGenCtx{}, field, good, good, good); }, std::invalid_argument);
}

TEST(gen_gf_div, divides_a_register_by_itself) {
    GF2Field field(4);
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());
    fuzzer.use_circuit_builder([&](CircuitBuilder &builder, std::mt19937_64 &) {
        auto Q_a = builder.append_register(4, "a");
        auto Q_target = builder.append_register(4, "target");
        auto Q_clean = builder.append_register(gf_div_workspace_size(field), "@clean");
        CircuitGenCtx ctx{.clean_workspace = Q_clean};
        // Dividing a register by itself (Q_lhs == Q_rhs) is valid when Q_target is disjoint.
        gen_gf_div(builder, ctx, field, Q_target, Q_a, Q_a);
    });
    fuzzer.use_output_sampler([&](OutputSample &sample) {
        GF2Poly a = GF2Poly::from_fixed_width_int(sample.old("a"));
        GF2Poly t = GF2Poly::from_fixed_width_int(sample.old("target"));
        sample["target"] = (t ^ field.div(a, a)).to_fixed_width_int(4);
    });
    fuzzer.fuzz(1, 64);
}

TEST(gen_gf_div, qubit_lifecycle_is_balanced) {
    // Division borrows both an inverse register and an Itoh-Tsujii chain register, so it is the
    // main consumer of the explicit chain overloads and the place a workspace leak would show up.
    for (size_t m : std::vector<size_t>{2, 3, 4, 5, 6, 7, 8, 9, 16}) {
        GF2Field field(m);
        for (bool undiv : std::vector<bool>{false, true}) {
            CircuitBuilder builder;
            auto lhs = builder.append_register(m, "lhs");
            auto rhs = builder.append_register(m, "rhs");
            auto target = builder.append_register(m, "target");
            auto clean = builder.append_register(gf_div_workspace_size(field), "@clean");
            builder.broadcast_reset(lhs);
            builder.broadcast_reset(rhs);
            builder.broadcast_reset(target);
            CircuitGenCtx ctx{.clean_workspace = clean};
            if (undiv) {
                gen_gf_undiv(builder, ctx, field, target, lhs, rhs);
            } else {
                gen_gf_div(builder, ctx, field, target, lhs, rhs);
                builder.broadcast_del_zero(target);
            }
            builder.broadcast_del_zero(lhs);
            builder.broadcast_del_zero(rhs);

            Circuit circuit = builder.finish_circuit();
            std::vector<bool> active(circuit.num_qubits, false);
            circuit.iter_ops([&](const Op &op) {
                if (op.kind == OpType::R) {
                    ASSERT_FALSE(active[op.q_target.untagged_id()]) << "m=" << m << " reset while active";
                    active[op.q_target.untagged_id()] = true;
                } else if (op.kind == OpType::HMR) {
                    ASSERT_TRUE(active[op.q_target.untagged_id()]) << "m=" << m << " measured out twice";
                    active[op.q_target.untagged_id()] = false;
                } else {
                    for (QubitOrFalse q : {op.q_target, op.q_control1, op.q_control2}) {
                        if (q.is_qubit()) {
                            ASSERT_TRUE(active[q.untagged_id()]) << "m=" << m << " used measured-out qubit";
                        }
                    }
                }
            });
            for (size_t q = 0; q < circuit.num_qubits; q++) {
                ASSERT_FALSE(active[q]) << "m=" << m << " qubit " << q << " never freed";
            }
        }
    }
}
