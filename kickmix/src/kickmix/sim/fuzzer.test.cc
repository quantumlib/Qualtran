#include "fuzzer.h"

#include "gtest/gtest.h"

#include "test.test.h"

using namespace kickmix;

TEST(fuzzer, simple) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t n = rng() % 10;
        auto Q_src = builder.append_register(n, "Q_src");
        auto Q_dst = builder.append_register(n, "Q_dst");
        builder.broadcast_cx(Q_src, Q_dst);
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        sample["Q_dst"] ^= sample["Q_src"];
    });

    fuzzer.fuzz(1, 256);
}

TEST(fuzzer, increment) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        auto qs = builder.append_register(3, "target");
        builder.ccx(qs[0], qs[1], qs[2]);
        builder.cx(qs[0], qs[1]);
        builder.x(qs[0]);
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        sample["target"].increment();
    });

    fuzzer.fuzz(1, 64);
}

TEST(fuzzer, bad_increment_0) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        auto qs = builder.append_register(3, "target");
        builder.ccx(qs[0], qs[1], qs[2]);
        builder.cx(qs[0], qs[1]);
        builder.x(qs[0]);
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        sample["target"].decrement();
    });

    EXPECT_THROW({ fuzzer.fuzz(1, 64); }, std::invalid_argument);
}

TEST(fuzzer, bad_increment_1) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        auto qs = builder.append_register(3, "target");
        builder.ccx(qs[0], qs[1], qs[2]);
        builder.cx(qs[0], qs[1]);
        builder.x(qs[0]);
        builder.x(qs[2]);
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        sample["target"].increment();
    });

    EXPECT_THROW({ fuzzer.fuzz(1, 64); }, std::invalid_argument);
}

TEST(fuzzer, bad_increment_2) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        auto qs = builder.append_register(3, "target");
        builder.ccx(qs[0], qs[1], qs[2]);
        builder.cx(qs[0], qs[1]);
        builder.x(qs[0]);
        builder.z(qs[2]);
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        sample["target"].increment();
    });

    EXPECT_THROW({ fuzzer.fuzz(1, 64); }, std::invalid_argument);
}

TEST(fuzzer, bad_increment_3) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        auto qs = builder.append_register(3, "target");
        builder.ccx(qs[0], qs[1], qs[2]);
        builder.cx(qs[0], qs[1]);
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        sample["target"].increment();
    });

    EXPECT_THROW({ fuzzer.fuzz(1, 64); }, std::invalid_argument);
}

TEST(fuzzer, z_good) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        auto q = builder.append_register(1, "target")[0];
        builder.z(q);
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        sample.phase_half_turns = sample["target"][0];
    });

    fuzzer.fuzz(1, 64);
}

TEST(fuzzer, z_bad) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        builder.append_register(1, "target");
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        sample.phase_half_turns = sample["target"][0];
    });

    EXPECT_THROW({ fuzzer.fuzz(1, 64); }, std::invalid_argument);
}
