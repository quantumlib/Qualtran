#include "gen_cmp.h"

#include <iostream>

#include "perf.perf.h"

using namespace kickmix;

BENCHMARK(gen_flip_if_lt__n256) {
    size_t data = 0;
    size_t num_operations = 0;
    CircuitBuilder builder;
    auto lhs = builder.append_register(256);
    auto rhs = builder.append_classical_register(256);
    auto target = builder.append_register(1)[0];
    auto clean = builder.reserve_qubits(256);
    CircuitGenCtx ctx{.clean_workspace = clean};
    benchmark_go([&]() {
        builder.mut.clear();
        gen_flip_if_lt(builder, ctx, lhs, rhs, target, false, true, INFINITY);
        data += builder.mut.qqq0.size();
        num_operations = builder.mut.op_types.size();
    })
        .goal_micros(1.3)
        .show_rate("ops", num_operations);

    if (!data) {
        std::cerr << "data dependence\n";
    }
}

BENCHMARK(gen_flip_if_lt__n256_3clean) {
    size_t data = 0;
    size_t num_operations = 0;
    CircuitBuilder builder;
    auto lhs = builder.append_register(256);
    auto rhs = builder.append_classical_register(256);
    auto target = builder.append_register(1)[0];
    auto clean = builder.reserve_qubits(3);
    CircuitGenCtx ctx{.clean_workspace = clean};
    benchmark_go([&]() {
        builder.mut.clear();
        gen_flip_if_lt(builder, ctx, lhs, rhs, target, false, true, INFINITY);
        data += builder.mut.qqq0.size();
        num_operations = builder.mut.op_types.size();
    })
        .goal_micros(3.5)
        .show_rate("ops", num_operations);

    if (!data) {
        std::cerr << "data dependence\n";
    }
}
