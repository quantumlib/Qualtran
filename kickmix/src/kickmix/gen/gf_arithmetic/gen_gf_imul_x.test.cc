#include "gen_gf_imul_x.h"

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

TEST(gen_gf_imul_x, mul_fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());
    auto field = std::make_shared<std::optional<GF2Field>>();
    auto power = std::make_shared<size_t>(0);

    fuzzer.use_circuit_builder([&](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t m = 1 + rng() % 9;
        *field = GF2Field(m);
        *power = rng() % (2 * m + 2);
        auto target = builder.append_register(m, "target");
        gen_gf_imul_x(builder, CircuitGenCtx{}, **field, target, *power);
    });

    fuzzer.use_output_sampler([&](OutputSample &sample) {
        FixedWidthInt &target = sample["target"];
        GF2Poly v = GF2Poly::from_fixed_width_int(target);
        write_poly(target, (*field)->mul(v, (*field)->pow((*field)->x(), *power)));
    });

    fuzzer.fuzz(60, 64);
}

TEST(gen_gf_imul_x, div_fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());
    auto field = std::make_shared<std::optional<GF2Field>>();
    auto power = std::make_shared<size_t>(0);

    fuzzer.use_circuit_builder([&](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t m = 1 + rng() % 9;
        *field = GF2Field(m);
        *power = rng() % (2 * m + 2);
        auto target = builder.append_register(m, "target");
        gen_gf_idiv_x(builder, CircuitGenCtx{}, **field, target, *power);
    });

    fuzzer.use_output_sampler([&](OutputSample &sample) {
        FixedWidthInt &target = sample["target"];
        GF2Poly v = GF2Poly::from_fixed_width_int(target);
        GF2Poly divisor = (*field)->pow((*field)->x(), *power);
        write_poly(target, (*field)->div(v, divisor));
    });

    fuzzer.fuzz(60, 64);
}

TEST(gen_gf_imul_x, div_undoes_mul_fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t m = 1 + rng() % 9;
        GF2Field field(m);
        size_t k = rng() % (3 * m + 2);
        auto target = builder.append_register(m, "target");
        gen_gf_imul_x(builder, CircuitGenCtx{}, field, target, k);
        gen_gf_idiv_x(builder, CircuitGenCtx{}, field, target, k);
    });

    fuzzer.fuzz(40, 64);
}

TEST(gen_gf_imul_x, shift_by_m_matches_field) {
    // k == m is the case the multiplier relies on, and the one the deferred rotation optimizes most.
    for (size_t m : std::vector<size_t>{2, 3, 4, 5, 6, 7, 8}) {
        GF2Field field(m);
        CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

        fuzzer.use_circuit_builder([&](CircuitBuilder &builder, std::mt19937_64 &rng) {
            auto target = builder.append_register(m, "target");
            gen_gf_imul_x(builder, CircuitGenCtx{}, field, target, m);
        });
        fuzzer.use_output_sampler([&](OutputSample &sample) {
            FixedWidthInt &target = sample["target"];
            GF2Poly v = GF2Poly::from_fixed_width_int(target);
            write_poly(target, field.mul(v, field.pow(field.x(), m)));
        });
        fuzzer.fuzz(1, 64);
    }
}

TEST(gen_gf_imul_x, large_power_uses_linear_map) {
    // A power big enough to trip the linear-map threshold must still be correct.
    size_t m = 6;
    GF2Field field(m);
    size_t k = m * m * 4;

    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());
    fuzzer.use_circuit_builder([&](CircuitBuilder &builder, std::mt19937_64 &rng) {
        auto target = builder.append_register(m, "target");
        gen_gf_imul_x(builder, CircuitGenCtx{}, field, target, k);
    });
    fuzzer.use_output_sampler([&](OutputSample &sample) {
        FixedWidthInt &target = sample["target"];
        GF2Poly v = GF2Poly::from_fixed_width_int(target);
        write_poly(target, field.mul(v, field.pow(field.x(), k)));
    });
    fuzzer.fuzz(1, 64);
}

TEST(gen_gf_imul_x, degenerate_cases) {
    GF2Field field(8);
    {
        // Shifting by zero is a no-op.
        CircuitBuilder builder;
        auto target = builder.append_register(8, "target");
        gen_gf_imul_x(builder, CircuitGenCtx{}, field, target, 0);
        gen_gf_idiv_x(builder, CircuitGenCtx{}, field, target, 0);
        ASSERT_EQ(builder.finish_circuit().num_ops, 0);
    }
    {
        // GF(2) has no element x, so multiplying by x is a no-op.
        GF2Field tiny(1);
        CircuitBuilder builder;
        auto target = builder.append_register(1, "target");
        gen_gf_imul_x(builder, CircuitGenCtx{}, tiny, target, 5);
        ASSERT_EQ(builder.finish_circuit().num_ops, 0);
    }
}

TEST(gen_gf_imul_x, rejects_bad_register_size) {
    GF2Field field(8);
    CircuitBuilder builder;
    auto target = builder.append_register(7, "target");
    ASSERT_THROW({ gen_gf_imul_x(builder, CircuitGenCtx{}, field, target, 1); }, std::invalid_argument);
    ASSERT_THROW({ gen_gf_idiv_x(builder, CircuitGenCtx{}, field, target, 1); }, std::invalid_argument);
}

TEST(gen_gf_imul_x, deferred_rotation_beats_repeated_rotation) {
    // Shifting by m must not cost anywhere near the m*(m-1) swaps that naive repeated rotation uses.
    size_t m = 16;
    GF2Field field(m);
    CircuitBuilder builder;
    auto target = builder.append_register(m, "target");
    gen_gf_imul_x(builder, CircuitGenCtx{}, field, target, m);
    size_t ops = builder.finish_circuit().num_ops;
    ASSERT_LT(ops, m * (m - 1) / 2) << ops;
}

TEST(gen_gf_imul_x, huge_k_no_overflow) {
    GF2Field field(8);
    size_t huge_k = (SIZE_MAX / 2) + 1;
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());
    fuzzer.use_circuit_builder([&](CircuitBuilder &builder, std::mt19937_64 &) {
        auto Q_target = builder.append_register(8, "target");
        gen_gf_imul_x(builder, {}, field, Q_target, huge_k);
    });
    GF2Poly factor = field.pow(field.x(), huge_k);
    fuzzer.use_output_sampler([&](OutputSample &sample) {
        GF2Poly v = GF2Poly::from_fixed_width_int(sample.old("target"));
        sample["target"] = field.mul(v, factor).to_fixed_width_int(8);
    });
    fuzzer.fuzz(1, 64);
}
