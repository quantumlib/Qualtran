#include "gen_gf_inverse.h"

#include "perf.perf.h"

using namespace kickmix;

BENCHMARK(gen_gf_inverse__m128) {
    GF2Field field(128);

    size_t num_operations = 0;
    CircuitBuilder builder;
    auto input = builder.append_register(128);
    auto target = builder.append_register(128);
    auto clean = builder.reserve_qubits(gf_inverse_workspace_size(field));
    CircuitGenCtx ctx{.clean_workspace = clean};

    benchmark_go([&]() {
        builder.mut.clear();
        gen_gf_inverse(builder, ctx, field, target, input);
        num_operations = builder.mut.op_types.size();
    })
        .goal_millis(25)
        .show_rate("ops", num_operations);
}

BENCHMARK(gen_gf_inverse_dirty__m128) {
    GF2Field field(128);

    size_t num_operations = 0;
    CircuitBuilder builder;
    auto input = builder.append_register(128);
    auto target = builder.append_register(128);
    auto chain = builder.append_register(gf_inverse_chain_size(field));
    CircuitGenCtx ctx{};

    benchmark_go([&]() {
        builder.mut.clear();
        gen_gf_inverse(builder, ctx, field, target, input, chain);
        num_operations = builder.mut.op_types.size();
    })
        .goal_millis(12)
        .show_rate("ops", num_operations);
}
