#include "gen_gf_iadd.h"

#include "gtest/gtest.h"

#include "kickmix/sim/fuzzer.h"
#include "test.test.h"

using namespace kickmix;

TEST(gen_gf_iadd, uncontrolled_fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t m = 1 + rng() % 12;
        auto target = builder.append_register(m, "target");
        auto offset = builder.append_register(m, "offset");
        gen_gf_iadd(builder, CircuitGenCtx{}, target, offset);
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        sample["target"] ^= sample["offset"];
    });

    fuzzer.fuzz(30, 64);
}

TEST(gen_gf_iadd, controlled_fuzz) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t m = 1 + rng() % 12;
        auto target = builder.append_register(m, "target");
        auto offset = builder.append_register(m, "offset");
        auto control = builder.append_register(1, "control")[0];
        gen_gf_iadd(builder, CircuitGenCtx{}, target, offset, control);
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        if ((bool)sample["control"]) {
            sample["target"] ^= sample["offset"];
        }
    });

    fuzzer.fuzz(30, 64);
}

TEST(gen_gf_iadd, is_self_inverse) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t m = 1 + rng() % 12;
        auto target = builder.append_register(m, "target");
        auto offset = builder.append_register(m, "offset");
        auto control = builder.append_register(1, "control")[0];
        gen_gf_iadd(builder, CircuitGenCtx{}, target, offset, control);
        gen_gf_iadd(builder, CircuitGenCtx{}, target, offset, control);
    });

    // Adding twice over GF(2) is the identity, so no output sampler is needed.
    fuzzer.fuzz(20, 64);
}

TEST(gen_gf_iadd, reversed_and_strided_views) {
    // Adding a register into its own reverse is a legitimate strided use of the API.
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        auto target = builder.append_register(6, "target");
        auto offset = builder.append_register(6, "offset");
        gen_gf_iadd(
            builder,
            CircuitGenCtx{},
            stride_span<const QubitId>(target),
            stride_span<const QubitId>(offset).reversed());
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        FixedWidthInt &target = sample["target"];
        const FixedWidthInt &offset = sample["offset"];
        size_t n = target.num_bits;
        for (size_t k = 0; k < n; k++) {
            target.bit_ref(k) = target.bit_ref(k) ^ ((offset.words[(n - 1 - k) / 64] >> ((n - 1 - k) & 63)) & 1);
        }
    });

    fuzzer.fuzz(1, 64);
}

TEST(gen_gf_iadd, empty_register) {
    CircuitBuilder builder;
    std::vector<QubitId> target;
    std::vector<QubitId> offset;
    gen_gf_iadd(builder, CircuitGenCtx{}, target, offset);
    ASSERT_EQ(builder.finish_circuit().num_ops, 0);
}

TEST(gen_gf_iadd, rejects_size_mismatch) {
    CircuitBuilder builder;
    auto target = builder.append_register(4, "target");
    auto offset = builder.append_register(5, "offset");
    ASSERT_THROW({ gen_gf_iadd(builder, CircuitGenCtx{}, target, offset); }, std::invalid_argument);
}
