#include "gen_gf_ifrobenius.h"

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

TEST(gen_gf_ifrobenius, fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());
    auto field = std::make_shared<std::optional<GF2Field>>();
    auto power = std::make_shared<size_t>(0);

    fuzzer.use_circuit_builder([&](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t m = 1 + rng() % 9;
        *field = GF2Field(m);
        *power = rng() % (2 * m + 2);
        auto target = builder.append_register(m, "target");
        gen_gf_ifrobenius(builder, CircuitGenCtx{}, **field, target, *power);
    });

    fuzzer.use_output_sampler([&](OutputSample &sample) {
        FixedWidthInt &target = sample["target"];
        GF2Poly v = GF2Poly::from_fixed_width_int(target);
        write_poly(target, (*field)->frobenius(v, *power));
    });

    fuzzer.fuzz(60, 64);
}

TEST(gen_gf_ifrobenius, adjoint_fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());
    auto field = std::make_shared<std::optional<GF2Field>>();
    auto power = std::make_shared<size_t>(0);

    fuzzer.use_circuit_builder([&](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t m = 1 + rng() % 9;
        *field = GF2Field(m);
        *power = rng() % (2 * m + 2);
        auto target = builder.append_register(m, "target");
        gen_gf_ifrobenius_adjoint(builder, CircuitGenCtx{}, **field, target, *power);
    });

    fuzzer.use_output_sampler([&](OutputSample &sample) {
        FixedWidthInt &target = sample["target"];
        GF2Poly v = GF2Poly::from_fixed_width_int(target);
        size_t m = (*field)->degree();
        // Undoing k squarings is the same as performing m - k more of them.
        write_poly(target, (*field)->frobenius(v, m - (*power % m)));
    });

    fuzzer.fuzz(60, 64);
}

TEST(gen_gf_ifrobenius, adjoint_undoes_forward_fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t m = 1 + rng() % 9;
        GF2Field field(m);
        size_t k = rng() % (3 * m + 2);
        auto target = builder.append_register(m, "target");
        gen_gf_ifrobenius(builder, CircuitGenCtx{}, field, target, k);
        gen_gf_ifrobenius_adjoint(builder, CircuitGenCtx{}, field, target, k);
    });

    fuzzer.fuzz(40, 64);
}

TEST(gen_gf_ifrobenius, matches_repeated_squaring) {
    // Squaring k times must agree with applying the single square circuit k times.
    size_t m = 7;
    GF2Field field(m);
    size_t k = 3;

    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());
    fuzzer.use_circuit_builder([&](CircuitBuilder &builder, std::mt19937_64 &rng) {
        auto target = builder.append_register(m, "target");
        gen_gf_ifrobenius(builder, CircuitGenCtx{}, field, target, k);
        for (size_t i = 0; i < k; i++) {
            gen_gf_ifrobenius_adjoint(builder, CircuitGenCtx{}, field, target, 1);
        }
    });
    fuzzer.fuzz(1, 64);
}

TEST(gen_gf_ifrobenius, degenerate_cases) {
    {
        // Squaring a multiple of m times is the identity.
        GF2Field field(8);
        CircuitBuilder builder;
        auto target = builder.append_register(8, "target");
        gen_gf_ifrobenius(builder, CircuitGenCtx{}, field, target, 0);
        gen_gf_ifrobenius(builder, CircuitGenCtx{}, field, target, 8);
        gen_gf_ifrobenius(builder, CircuitGenCtx{}, field, target, 24);
        gen_gf_ifrobenius_adjoint(builder, CircuitGenCtx{}, field, target, 16);
        ASSERT_EQ(builder.finish_circuit().num_ops, 0);
    }
    {
        // Every element of GF(2) is its own square.
        GF2Field tiny(1);
        CircuitBuilder builder;
        auto target = builder.append_register(1, "target");
        gen_gf_ifrobenius(builder, CircuitGenCtx{}, tiny, target, 5);
        ASSERT_EQ(builder.finish_circuit().num_ops, 0);
    }
}

TEST(gen_gf_ifrobenius, rejects_bad_register_size) {
    GF2Field field(8);
    CircuitBuilder builder;
    auto target = builder.append_register(7, "target");
    ASSERT_THROW({ gen_gf_ifrobenius(builder, CircuitGenCtx{}, field, target, 1); }, std::invalid_argument);
    ASSERT_THROW({ gen_gf_ifrobenius_adjoint(builder, CircuitGenCtx{}, field, target, 1); }, std::invalid_argument);
}

TEST(gen_gf_ifrobenius, large_degree_is_linear_and_ancilla_free) {
    // The matrix construction must not overflow for k >= 64, which the original 64 bit code did.
    GF2Field field(128);
    CircuitBuilder builder;
    auto target = builder.append_register(128, "target");
    gen_gf_ifrobenius(builder, CircuitGenCtx{}, field, target, 100);
    Circuit circuit = builder.finish_circuit();
    ASSERT_EQ(circuit.num_qubits, 128);
    ASSERT_EQ(circuit.max_magic(), 0);

    // Sanity check the classical map it is synthesized from: it must be an order m automorphism.
    GF2Poly v = GF2Poly::from_str("0x1234567890abcdef1234567890abcdef");
    field.ireduce(v);
    ASSERT_EQ(field.frobenius(v, 128), v);
    ASSERT_EQ(field.frobenius(v, 1), field.square(v));
}
