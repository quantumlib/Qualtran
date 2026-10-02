#include "gen_cmp.h"

#include <fstream>

#include "gtest/gtest.h"

#include "gen_cmp_low_space.h"
#include "kickmix/sim/fuzzer.h"
#include "kickmix/util/circuit_testing.test.h"
#include "test.test.h"

using namespace kickmix;

TEST(icmp, cost) {
    for (size_t n = 2; n < 20; n++) {
        CircuitBuilder builder;
        auto lhs = builder.append_register(n);
        auto rhs = builder.append_register_qcarray_result(n);
        auto target = builder.append_register(1)[0];
        auto or_equal = builder.append_register_qcarray_result(1)[0];
        auto control = builder.append_register(1)[0];
        auto clean = builder.reserve_qubits(n);
        gen_flip_if_lt(builder, CircuitGenCtx{.clean_workspace = clean}, lhs, rhs, target, or_equal, control, INFINITY);
        auto circuit = builder.finish_circuit();
        EXPECT_LE(circuit.max_magic(), n + 2) << n;
    }
}

TEST(icmp, cost_low_workspace) {
    for (size_t n = 3; n < 20; n++) {
        CircuitBuilder builder;
        auto lhs = builder.append_register(n);
        auto rhs = builder.append_classical_register_qcarray_result(n);
        auto target = builder.append_register(1)[0];
        auto or_equal = builder.append_register_qcarray_result(1)[0];
        auto control = builder.append_register(1)[0];
        auto clean = builder.reserve_qubits(5);
        gen_flip_if_lt(builder, CircuitGenCtx{.clean_workspace = clean}, lhs, rhs, target, or_equal, control, INFINITY);
        auto circuit = builder.finish_circuit();
        EXPECT_LE(circuit.max_magic(), 3 * n) << n;
    }
}

TEST(icmp, fuzz_lt_phase) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t n = rng() % 8;
        auto lhs = builder.append_register(n, "lhs");
        auto rhs = builder.append_classical_register_qcarray_result(n, "rhs");
        auto clean = builder.append_register(n, "@clean");
        auto or_equal = builder.append_register(1, "or_equal")[0];
        auto control = builder.append_register(1, "control")[0];
        CircuitGenCtx ctx{.clean_workspace = clean};
        gen_flip_if_lt(builder, ctx, lhs, rhs, MINUS_KET, or_equal, control, INFINITY);
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        if (sample["control"]) {
            if (sample["or_equal"]) {
                sample.phase_half_turns = sample["lhs"] <= sample["rhs"];
            } else {
                sample.phase_half_turns = sample["lhs"] < sample["rhs"];
            }
        }
    });

    fuzzer.fuzz(10, 64);
}

struct BackedQubitOrBitOrBoolRegister {
    array_z rhs;
    BackedQubitOrBitOrBoolRegister(size_t n, CircuitBuilder &builder, std::mt19937_64 &rng, std::string_view name) {
        size_t type = rng() % 3;
        if (type == 0) {
            rhs = builder.append_register_qcarray_result(n, name);
        } else if (type == 1) {
            rhs = builder.append_classical_register_qcarray_result(n, name);
        } else {
            RegisterId r{builder.next_register_id};
            auto rhs_mixed = builder.append_classical_register_mixed_result(n >> 1, name);
            auto qs = builder.reserve_qubits(n - rhs_mixed.size());
            builder.append_qubit_to_register(qs, r);
            for (auto e : qs) {
                rhs_mixed.push_back(e);
            }
            rhs = array_z::copy_of(rhs_mixed);
        }
    }
};

TEST(icmp, fuzz_lt) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t n = rng() % 10;
        auto lhs = builder.append_register(n, "lhs");
        BackedQubitOrBitOrBoolRegister rhs(n, builder, rng, "rhs");
        auto target = builder.append_register(1, "target")[0];
        auto or_equal = builder.append_classical_register_qcarray_result(1, "or_equal")[0];
        auto control = builder.append_register(1, "control")[0];
        auto clean = builder.append_register(n, "@clean");
        CircuitGenCtx ctx{.clean_workspace = clean};
        gen_flip_if_lt(builder, ctx, lhs, rhs.rhs, target, or_equal, control, INFINITY);
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

TEST(icmp, fuzz_lt_low_space) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t n = rng() % 30;
        auto lhs = builder.append_register(n, "lhs");
        auto rhs = builder.append_classical_register(n, "rhs");
        auto target = builder.append_register(1, "target")[0];
        auto or_equal = builder.append_classical_register_qcarray_result(1, "or_equal")[0];
        auto control = builder.append_register(1, "control")[0];
        auto clean = builder.append_register(3, "@clean");
        CircuitGenCtx ctx{.clean_workspace = clean};
        gen_flip_if_lt(builder, ctx, lhs, rhs, target, or_equal, control, INFINITY);
    });

    fuzzer.use_input_sampler([](InputSample &sample, std::mt19937_64 &rng) {
        sample["control"].randomize(rng);
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

TEST(icmp, fuzz_gt) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t n = rng() % 32;
        auto lhs = builder.append_register(n, "lhs");
        auto rhs = builder.append_classical_register_qcarray_result(n, "rhs");
        auto target = builder.append_register(1, "target")[0];
        auto control = builder.append_register(1, "control")[0];
        auto or_equal = builder.append_classical_register_qcarray_result(1, "or_equal")[0];
        auto clean = builder.append_register(n, "@clean");
        CircuitGenCtx ctx{.clean_workspace = clean};
        gen_flip_if_gt(builder, ctx, lhs, rhs, target, or_equal, control, INFINITY);
    });

    fuzzer.use_input_sampler([](InputSample &sample, std::mt19937_64 &rng) {
        sample["lhs"].randomize(rng);
        sample["target"].randomize(rng);
        sample["control"].randomize(rng);
        sample["or_equal"].randomize(rng);
        if (rng() % 2) {
            // Give a case near the threshold.
            sample["rhs"] = sample["lhs"];
            sample["rhs"] += (int)(rng() % 20) - 10;
        } else {
            sample["rhs"].randomize(rng);
        }
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        if (sample["control"]) {
            if (sample["or_equal"]) {
                sample["target"] ^= sample["lhs"] >= sample["rhs"];
            } else {
                sample["target"] ^= sample["lhs"] > sample["rhs"];
            }
        }
    });

    fuzzer.fuzz(10, 64);
}

TEST(icmp, fuzz_le) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t n = rng() % 32;
        auto lhs = builder.append_register(n, "lhs");
        auto rhs = builder.append_classical_register_qcarray_result(n, "rhs");
        auto target = builder.append_register(1, "target")[0];
        auto control = builder.append_register(1, "control")[0];
        auto clean = builder.append_register(n, "@clean");
        CircuitGenCtx ctx{.clean_workspace = clean};
        gen_flip_if_le(builder, ctx, lhs, rhs, target, control, INFINITY);
    });

    fuzzer.use_input_sampler([](InputSample &sample, std::mt19937_64 &rng) {
        sample["lhs"].randomize(rng);
        sample["target"].randomize(rng);
        sample["control"].randomize(rng);
        if (rng() % 2) {
            // Give a case near the threshold.
            sample["rhs"] = sample["lhs"];
            sample["rhs"] += (int)(rng() % 20) - 10;
        } else {
            sample["rhs"].randomize(rng);
        }
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        if (sample["control"]) {
            sample["target"] ^= sample["lhs"] <= sample["rhs"];
        }
    });

    fuzzer.fuzz(10, 64);
}

TEST(icmp, fuzz_ge) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t n = rng() % 32;
        auto lhs = builder.append_register(n, "lhs");
        auto rhs = builder.append_classical_register_qcarray_result(n, "rhs");
        auto target = builder.append_register(1, "target")[0];
        auto control = builder.append_register(1, "control")[0];
        auto clean = builder.append_register(n, "@clean");
        CircuitGenCtx ctx{.clean_workspace = clean};
        gen_flip_if_ge(builder, ctx, lhs, rhs, target, control, INFINITY);
    });

    fuzzer.use_input_sampler([](InputSample &sample, std::mt19937_64 &rng) {
        sample["lhs"].randomize(rng);
        sample["target"].randomize(rng);
        sample["control"].randomize(rng);
        if (rng() % 2) {
            // Give a case near the threshold.
            sample["rhs"] = sample["lhs"];
            sample["rhs"] += (int)(rng() % 20) - 10;
        } else {
            sample["rhs"].randomize(rng);
        }
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        if (sample["control"]) {
            sample["target"] ^= sample["lhs"] >= sample["rhs"];
        }
    });

    fuzzer.fuzz(10, 64);
}

TEST(icmp, fuzz_eq) {
    CircuitFuzzer fuzzer(INDEPENDENT_TEST_RNG());

    fuzzer.use_circuit_builder([](CircuitBuilder &builder, std::mt19937_64 &rng) {
        size_t n = rng() % 32;
        auto lhs = builder.append_register(n, "lhs");
        auto rhs = builder.append_classical_register_qcarray_result(n, "rhs");
        auto target = builder.append_register(1, "target")[0];
        auto control = builder.append_register(1, "control")[0];
        auto clean = builder.append_register(n, "@clean");
        CircuitGenCtx ctx{.clean_workspace = clean};
        gen_flip_if_eq(builder, ctx, lhs, rhs, target, control);
    });

    fuzzer.use_input_sampler([](InputSample &sample, std::mt19937_64 &rng) {
        sample["lhs"].randomize(rng);
        sample["target"].randomize(rng);
        sample["control"].randomize(rng);
        if (rng() % 2) {
            // Give a case near the threshold.
            sample["rhs"] = sample["lhs"];
            sample["rhs"] += (int)(rng() % 10) - 5;
        } else {
            sample["rhs"].randomize(rng);
        }
    });

    fuzzer.use_output_sampler([](OutputSample &sample) {
        if (sample["control"]) {
            sample["target"] ^= sample["lhs"] == sample["rhs"];
        }
    });

    fuzzer.fuzz(10, 64);
}
