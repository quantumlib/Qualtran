#include "gen_gf_imul_classical.h"

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

/// Picks a uniformly random non-zero element of the field.
static GF2Poly random_nonzero_element(const GF2Field &field, std::mt19937_64 &rng) {
    GF2Poly result;
    while (result.is_zero()) {
        for (size_t k = 0; k < field.degree(); k++) {
            result.set_bit(k, rng() & 1);
        }
    }
    return result;
}

TEST(gen_gf_imul_classical, mul_fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());
    auto field = std::make_shared<std::optional<GF2Field>>();
    auto constant = std::make_shared<GF2Poly>();

    fuzzer.use_circuit_builder([&](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t m = 1 + rng() % 9;
        *field = GF2Field(m);
        *constant = random_nonzero_element(**field, rng);
        auto target = builder.append_register(m, "target");
        gen_gf_imul_classical(builder, CircuitGenCtx{}, **field, target, *constant);
    });

    fuzzer.use_output_sampler([&](OutputSample &sample) {
        FixedWidthInt &target = sample["target"];
        GF2Poly v = GF2Poly::from_fixed_width_int(target);
        write_poly(target, (*field)->mul(v, *constant));
    });

    fuzzer.fuzz(60, 64);
}

TEST(gen_gf_imul_classical, div_fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());
    auto field = std::make_shared<std::optional<GF2Field>>();
    auto constant = std::make_shared<GF2Poly>();

    fuzzer.use_circuit_builder([&](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t m = 1 + rng() % 9;
        *field = GF2Field(m);
        *constant = random_nonzero_element(**field, rng);
        auto target = builder.append_register(m, "target");
        gen_gf_idiv_classical(builder, CircuitGenCtx{}, **field, target, *constant);
    });

    fuzzer.use_output_sampler([&](OutputSample &sample) {
        FixedWidthInt &target = sample["target"];
        GF2Poly v = GF2Poly::from_fixed_width_int(target);
        write_poly(target, (*field)->div(v, *constant));
    });

    fuzzer.fuzz(60, 64);
}

TEST(gen_gf_imul_classical, div_undoes_mul_fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t m = 1 + rng() % 9;
        GF2Field field(m);
        GF2Poly constant = random_nonzero_element(field, rng);
        auto target = builder.append_register(m, "target");
        gen_gf_imul_classical(builder, CircuitGenCtx{}, field, target, constant);
        gen_gf_idiv_classical(builder, CircuitGenCtx{}, field, target, constant);
    });

    fuzzer.fuzz(40, 64);
}

TEST(gen_gf_imul_classical, unreduced_constant_is_accepted) {
    // Constants of degree >= m are reduced into the field instead of rejected.
    size_t m = 5;
    GF2Field field(m);
    GF2Poly raw = GF2Poly::from_str("0b101101011");

    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());
    fuzzer.use_circuit_builder([&](CircuitBuilder &builder, std::mt19937_64 &rng) {
        auto target = builder.append_register(m, "target");
        gen_gf_imul_classical(builder, CircuitGenCtx{}, field, target, raw);
    });
    fuzzer.use_output_sampler([&](OutputSample &sample) {
        FixedWidthInt &target = sample["target"];
        GF2Poly v = GF2Poly::from_fixed_width_int(target);
        write_poly(target, field.mul(v, field.mod(raw)));
    });
    fuzzer.fuzz(1, 64);
}

TEST(gen_gf_imul_classical, multiplying_by_one_is_free) {
    GF2Field field(8);
    CircuitBuilder builder;
    auto target = builder.append_register(8, "target");
    gen_gf_imul_classical(builder, CircuitGenCtx{}, field, target, field.one());
    gen_gf_idiv_classical(builder, CircuitGenCtx{}, field, target, field.one());
    ASSERT_EQ(builder.finish_circuit().num_ops, 0);
}

TEST(gen_gf_imul_classical, rejects_bad_arguments) {
    GF2Field field(8);
    {
        CircuitBuilder builder;
        auto target = builder.append_register(7, "target");
        ASSERT_THROW(
            { gen_gf_imul_classical(builder, CircuitGenCtx{}, field, target, field.one()); }, std::invalid_argument);
        ASSERT_THROW(
            { gen_gf_idiv_classical(builder, CircuitGenCtx{}, field, target, field.one()); }, std::invalid_argument);
    }
    {
        CircuitBuilder builder;
        auto target = builder.append_register(8, "target");
        ASSERT_THROW(
            { gen_gf_imul_classical(builder, CircuitGenCtx{}, field, target, field.zero()); }, std::invalid_argument);
        ASSERT_THROW(
            { gen_gf_idiv_classical(builder, CircuitGenCtx{}, field, target, field.zero()); }, std::invalid_argument);
        // A non-zero polynomial that reduces to zero is still zero.
        ASSERT_THROW(
            { gen_gf_imul_classical(builder, CircuitGenCtx{}, field, target, field.modulus()); },
            std::invalid_argument);
    }
}

TEST(gen_gf_imul_classical, uses_no_ancillas) {
    GF2Field field(16);
    CircuitBuilder builder;
    auto target = builder.append_register(16, "target");
    gen_gf_imul_classical(builder, CircuitGenCtx{}, field, target, GF2Poly::from_u64(0b1011010110101101));
    Circuit circuit = builder.finish_circuit();
    ASSERT_EQ(circuit.num_qubits, 16);
    ASSERT_EQ(circuit.max_magic(), 0);
}
