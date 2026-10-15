#include "gen_gf_mul.h"

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

TEST(gen_gf_mul, poly_mul_fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());
    auto size = std::make_shared<size_t>(0);

    fuzzer.use_circuit_builder([&](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t n = 1 + rng() % 8;
        *size = n;
        auto target = builder.append_register(2 * n - 1, "target");
        auto lhs = builder.append_register(n, "lhs");
        auto rhs = builder.append_register(n, "rhs");
        gen_gf2_poly_mul(builder, CircuitGenCtx{}, target, lhs, rhs);
    });

    fuzzer.use_output_sampler([&](OutputSample &sample) {
        GF2Poly lhs = GF2Poly::from_fixed_width_int(sample["lhs"]);
        GF2Poly rhs = GF2Poly::from_fixed_width_int(sample["rhs"]);
        FixedWidthInt &target = sample["target"];
        write_poly(target, GF2Poly::from_fixed_width_int(target) ^ GF2Poly::mul(lhs, rhs));
    });

    fuzzer.fuzz(80, 64);
}

TEST(gen_gf_mul, poly_mul_uses_no_ancillas_and_beats_schoolbook) {
    for (size_t n : std::vector<size_t>{4, 8, 16, 32}) {
        CircuitBuilder builder;
        auto target = builder.append_register(2 * n - 1);
        auto lhs = builder.append_register(n);
        auto rhs = builder.append_register(n);
        gen_gf2_poly_mul(builder, CircuitGenCtx{}, target, lhs, rhs);
        Circuit circuit = builder.finish_circuit();
        ASSERT_EQ(circuit.num_qubits, 4 * n - 1) << n;
        // Karatsuba uses 3^ceil(log2(n)) Toffolis, which is well under the schoolbook n^2.
        ASSERT_LT(circuit.max_magic(), n * n) << n;
    }
}

TEST(gen_gf_mul, mul_fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());
    auto field = std::make_shared<std::optional<GF2Field>>();

    fuzzer.use_circuit_builder([&](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t m = 1 + rng() % 9;
        *field = GF2Field(m);
        auto target = builder.append_register(m, "target");
        auto lhs = builder.append_register(m, "lhs");
        auto rhs = builder.append_register(m, "rhs");
        gen_gf_mul(builder, CircuitGenCtx{}, **field, target, lhs, rhs);
    });

    fuzzer.use_output_sampler([&](OutputSample &sample) {
        GF2Poly lhs = GF2Poly::from_fixed_width_int(sample["lhs"]);
        GF2Poly rhs = GF2Poly::from_fixed_width_int(sample["rhs"]);
        FixedWidthInt &target = sample["target"];
        GF2Poly v = GF2Poly::from_fixed_width_int(target);
        write_poly(target, v ^ (*field)->mul(lhs, rhs));
    });

    fuzzer.fuzz(80, 64);
}

TEST(gen_gf_mul, mul_uncontrolled_needs_no_workspace) {
    // The Karatsuba decomposition is entirely in place, so an uncontrolled product must not touch
    // the workspace at all.
    GF2Field field(16);
    CircuitBuilder builder;
    auto target = builder.append_register(16);
    auto lhs = builder.append_register(16);
    auto rhs = builder.append_register(16);
    gen_gf_mul(builder, CircuitGenCtx{}, field, target, lhs, rhs);
    ASSERT_EQ(builder.finish_circuit().num_qubits, 48);
}

TEST(gen_gf_mul, controlled_mul_fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());
    auto field = std::make_shared<std::optional<GF2Field>>();

    fuzzer.use_circuit_builder([&](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t m = 1 + rng() % 7;
        *field = GF2Field(m);
        auto target = builder.append_register(m, "target");
        auto lhs = builder.append_register(m, "lhs");
        auto rhs = builder.append_register(m, "rhs");
        auto control = builder.append_register(1, "control");
        auto clean = builder.append_register(m, "@clean");
        gen_gf_mul(builder, CircuitGenCtx{.clean_workspace = clean}, **field, target, lhs, rhs, control[0]);
    });

    fuzzer.use_output_sampler([&](OutputSample &sample) {
        if (!sample["control"].bit_ref(0)) {
            return;
        }
        GF2Poly lhs = GF2Poly::from_fixed_width_int(sample["lhs"]);
        GF2Poly rhs = GF2Poly::from_fixed_width_int(sample["rhs"]);
        FixedWidthInt &target = sample["target"];
        GF2Poly v = GF2Poly::from_fixed_width_int(target);
        write_poly(target, v ^ (*field)->mul(lhs, rhs));
    });

    fuzzer.fuzz(60, 64);
}

TEST(gen_gf_mul, unmul_undoes_mul_fuzz) {
    // The product register is declared clean, so the fuzzer asserts it comes back to zero.
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t m = 1 + rng() % 9;
        GF2Field field(m);
        auto lhs = builder.append_register(m, "lhs");
        auto rhs = builder.append_register(m, "rhs");
        auto product = builder.append_register(m, "@clean");
        gen_gf_mul(builder, CircuitGenCtx{}, field, product, lhs, rhs);
        gen_gf_unmul(builder, CircuitGenCtx{}, field, product, lhs, rhs);
    });

    fuzzer.fuzz(60, 64);
}

TEST(gen_gf_mul, unmul_is_toffoli_free) {
    GF2Field field(16);
    CircuitBuilder builder;
    auto product = builder.append_register(16);
    auto lhs = builder.append_register(16);
    auto rhs = builder.append_register(16);
    gen_gf_unmul(builder, CircuitGenCtx{}, field, product, lhs, rhs);
    ASSERT_EQ(builder.finish_circuit().max_magic(), 0);
}

TEST(gen_gf_mul, controlled_unmul_undoes_controlled_mul_fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t m = 1 + rng() % 7;
        GF2Field field(m);
        auto lhs = builder.append_register(m, "lhs");
        auto rhs = builder.append_register(m, "rhs");
        auto target = builder.append_register(m, "@dirty");
        auto control = builder.append_register(1, "control");
        auto clean = builder.append_register(m, "@clean");
        CircuitGenCtx ctx{.clean_workspace = clean};
        gen_gf_mul(builder, ctx, field, target, lhs, rhs, control[0]);
        gen_gf_unmul(builder, ctx, field, target, lhs, rhs, control[0]);
    });

    fuzzer.fuzz(40, 64);
}

TEST(gen_gf_mul, phase_by_product_matches_product_parity_fuzz) {
    // Phasing by a mask, then uncomputing a product that carries the same mask, must cancel out.
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t m = 1 + rng() % 8;
        GF2Field field(m);
        auto lhs = builder.append_register(m, "lhs");
        auto rhs = builder.append_register(m, "rhs");
        auto product = builder.append_register(m, "@clean");
        gen_gf_mul(builder, CircuitGenCtx{}, field, product, lhs, rhs);

        // Measure the product out by hand, then pay back the phase with the exposed primitive.
        std::vector<CircuitBuilderRaiiXBit> measured;
        std::vector<BitId> mask;
        for (size_t i = 0; i < m; i++) {
            measured.push_back(builder.hmr_raii_xbit(product[i]));
            mask.push_back(measured.back().bit);
        }
        gen_gf_phase_by_product(builder, CircuitGenCtx{}, field, mask, lhs, rhs);
    });

    fuzzer.fuzz(60, 64);
}

TEST(gen_gf_mul, large_degree_is_buildable) {
    // Building at the maximum supported degree (512) must work, and
    // must stay far below the m^2 Toffolis a schoolbook product would need.
    GF2Field field(512);
    CircuitBuilder builder;
    auto target = builder.append_register(512);
    auto lhs = builder.append_register(512);
    auto rhs = builder.append_register(512);
    gen_gf_mul(builder, CircuitGenCtx{}, field, target, lhs, rhs);
    Circuit circuit = builder.finish_circuit();
    ASSERT_EQ(circuit.num_qubits, 3 * 512);
    ASSERT_LT(circuit.max_magic(), 512 * 512 / 4);
}

TEST(gen_gf_mul, toffoli_gate_counts) {
    // T(1) = 1; T(m) = 2 * T(ceil(m / 2)) + T(floor(m / 2)).
    std::vector<std::pair<size_t, size_t>> expected_counts = {
        {10, 51},
        {12, 63},
        {64, 729},
        {128, 2187},
        {256, 6561},
        {512, 19683},
    };
    for (auto [m, expected_toffolis] : expected_counts) {
        GF2Field field(m);
        CircuitBuilder builder;
        auto target = builder.append_register(m);
        auto lhs = builder.append_register(m);
        auto rhs = builder.append_register(m);
        gen_gf_mul(builder, CircuitGenCtx{}, field, target, lhs, rhs);
        Circuit circuit = builder.finish_circuit();
        ASSERT_EQ(circuit.max_magic(), expected_toffolis) << "m=" << m;
    }
}

TEST(gen_gf_mul, rejects_bad_register_sizes) {
    GF2Field field(8);
    CircuitBuilder builder;
    auto good = builder.append_register(8);
    auto bad = builder.append_register(7);
    ASSERT_THROW({ gen_gf_mul(builder, CircuitGenCtx{}, field, bad, good, good); }, std::invalid_argument);
    ASSERT_THROW({ gen_gf_mul(builder, CircuitGenCtx{}, field, good, bad, good); }, std::invalid_argument);
    ASSERT_THROW({ gen_gf_mul(builder, CircuitGenCtx{}, field, good, good, bad); }, std::invalid_argument);
    ASSERT_THROW({ gen_gf_unmul(builder, CircuitGenCtx{}, field, bad, good, good); }, std::invalid_argument);
    ASSERT_THROW({ gen_gf2_poly_mul(builder, CircuitGenCtx{}, good, good, good); }, std::invalid_argument);
    // A controlled product without enough clean workspace must complain instead of silently
    // reaching past the end of the workspace.
    ASSERT_THROW({ gen_gf_mul(builder, CircuitGenCtx{}, field, good, good, good, good[0]); }, std::invalid_argument);
}
