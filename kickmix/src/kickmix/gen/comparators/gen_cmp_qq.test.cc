#include "gen_cmp_qq.h"

#include "gtest/gtest.h"

#include "kickmix/sim/fuzzer.h"
#include "kickmix/util/circuit_testing.test.h"
#include "test.test.h"

using namespace kickmix;

TEST(gen_cmp_qq, fuzz_quantum_or_equal) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t n = rng() % 20;
        auto lhs = builder.append_register(n, "lhs");
        auto rhs = builder.append_register(n, "rhs");
        auto target = builder.append_register(1, "target")[0];
        auto or_equal = builder.append_register(1, "or_equal")[0];
        auto clean = builder.append_register(rng() % 20, "@clean");
        auto control = builder.append_register(1, "control")[0];
        CircuitGenCtx ctx{.clean_workspace = clean};
        gen_flip_if_lt_qq(builder, ctx, lhs, rhs, target, or_equal, control);
    });

    fuzzer.use_input_sampler([](InputSample &sample, std::mt19937_64 &rng) {
        sample["control"].set_bit(0, rng() % 4 < 3);
        sample["lhs"].randomize(rng);
        sample["or_equal"].randomize(rng);
        sample["target"].randomize(rng);
        if (rng() % 2) {
            // Give a near-value case to check <= vs < more thoroughly at large sizes.
            sample["rhs"] = sample["lhs"];
            sample["rhs"] += (int)(rng() % 20) - 10;
        } else {
            sample["rhs"].randomize(rng);
        }
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        if (sample["control"]) {
            if (sample["or_equal"]) {
                sample["target"] ^= sample["lhs"] <= sample["rhs"];
            } else {
                sample["target"] ^= sample["lhs"] < sample["rhs"];
            }
        }
    });

    fuzzer.fuzz(100, 256);
}

TEST(gen_cmp_qq, cost) {
    for (size_t n = 5; n < 10; n++) {
        for (size_t c = 1; c < 10; c++) {
            CircuitBuilder builder;
            auto lhs = builder.append_register(n, "lhs");
            auto rhs = builder.append_register(n, "rhs");
            auto target = builder.append_register(1, "target")[0];
            auto or_equal = builder.append_classical_register(1, "or_equal")[0];
            auto clean = builder.append_register(c, "@clean");
            QubitOrTrue control = true;
            CircuitGenCtx ctx{.clean_workspace = clean};
            gen_flip_if_lt_qq(builder, ctx, lhs, rhs, target, or_equal, control);
            auto circuit = builder.finish_circuit();
            auto expected = std::max((int)n, 2 * (int)n - std::max((int)c - 1, 1));
            auto actual = circuit.max_magic();
            EXPECT_EQ(actual, expected) << n << ", " << c;
        }
    }
}
TEST(gen_cmp_qq, fuzz_classical_or_equal) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t n = rng() % 10;
        auto lhs = builder.append_register(n, "lhs");
        auto rhs = builder.append_register(n, "rhs");
        auto target = builder.append_register(1, "target")[0];
        auto or_equal = builder.append_classical_register(1, "or_equal")[0];
        auto clean = builder.append_register(1 + rng() % 20, "@clean");
        auto control = builder.append_register(1, "control")[0];
        CircuitGenCtx ctx{.clean_workspace = clean};
        gen_flip_if_lt_qq(builder, ctx, lhs, rhs, target, or_equal, control);
    });

    fuzzer.use_input_sampler([](InputSample &sample, std::mt19937_64 &rng) {
        sample["control"].set_bit(0, rng() % 4 < 3);
        sample["lhs"].randomize(rng);
        sample["or_equal"].randomize(rng);
        sample["target"].randomize(rng);
        if (rng() % 2) {
            // Give a near-value case to check <= vs < more thoroughly at large sizes.
            sample["rhs"] = sample["lhs"];
            sample["rhs"] += (int)(rng() % 20) - 10;
        } else {
            sample["rhs"].randomize(rng);
        }
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        if (sample["control"]) {
            if (sample["or_equal"]) {
                sample["target"] ^= sample["lhs"] <= sample["rhs"];
            } else {
                sample["target"] ^= sample["lhs"] < sample["rhs"];
            }
        }
    });

    fuzzer.fuzz(100, 256);
}
