#include "gen_gf_div.h"

#include "perf.perf.h"

using namespace kickmix;

BENCHMARK(gen_gf_div__m128) {
    GF2Field field(128);

    size_t num_operations = 0;
    CircuitBuilder builder;
    auto lhs = builder.append_register(128);
    auto rhs = builder.append_register(128);
    auto target = builder.append_register(128);
    auto clean = builder.reserve_qubits(gf_div_workspace_size(field));
    CircuitGenCtx ctx{.clean_workspace = clean};

    benchmark_go([&]() {
        builder.mut.clear();
        gen_gf_div(builder, ctx, field, target, lhs, rhs);
        num_operations = builder.mut.op_types.size();
    })
        .goal_millis(25)
        .show_rate("ops", num_operations);
}

BENCHMARK(gen_gf_undiv__m128) {
    GF2Field field(128);

    size_t num_operations = 0;
    CircuitBuilder builder;
    auto lhs = builder.append_register(128);
    auto rhs = builder.append_register(128);
    auto target = builder.append_register(128);
    auto clean = builder.reserve_qubits(gf_div_workspace_size(field));
    CircuitGenCtx ctx{.clean_workspace = clean};

    benchmark_go([&]() {
        builder.mut.clear();
        gen_gf_undiv(builder, ctx, field, target, lhs, rhs);
        num_operations = builder.mut.op_types.size();
    })
        .goal_millis(25)
        .show_rate("ops", num_operations);
}
