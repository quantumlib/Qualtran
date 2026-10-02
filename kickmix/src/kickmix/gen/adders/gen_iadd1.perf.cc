#include "gen_iadd1.h"

#include <iostream>

#include "perf.perf.h"

using namespace kickmix;

BENCHMARK(gen_iadd1__n256) {
    size_t data = 0;
    size_t num_operations = 0;
    CircuitBuilder builder;
    std::vector<QubitId> target = builder.append_register(256);
    auto clean = builder.reserve_qubits(256);
    CircuitGenCtx ctx{.clean_workspace = clean};
    ctx.minimize_qubits = false;
    benchmark_go([&]() {
        builder.mut.clear();
        gen_iadd1(builder, ctx, target);
        data += builder.mut.qqq0.size();
        num_operations = builder.mut.op_types.size();
    })
        .goal_nanos(580)
        .show_rate("ops", num_operations);

    if (!data) {
        std::cerr << "data dependence\n";
    }
}

BENCHMARK(gen_iadd1_control__n256) {
    size_t data = 0;
    size_t num_operations = 0;
    CircuitBuilder builder;
    std::vector<QubitId> target = builder.append_register(256);
    QubitId control = builder.append_register(1)[0];
    auto clean = builder.reserve_qubits(256);
    CircuitGenCtx ctx{.clean_workspace = clean};
    ctx.minimize_qubits = false;
    benchmark_go([&]() {
        builder.mut.clear();
        gen_iadd1(builder, ctx, target, control);
        data += builder.mut.qqq0.size();
        num_operations = builder.mut.op_types.size();
    })
        .goal_nanos(610)
        .show_rate("ops", num_operations);

    if (!data) {
        std::cerr << "data dependence\n";
    }
}
