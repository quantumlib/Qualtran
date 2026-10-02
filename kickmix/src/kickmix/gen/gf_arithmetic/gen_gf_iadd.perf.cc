#include "gen_gf_iadd.h"

#include "perf.perf.h"

using namespace kickmix;

BENCHMARK(gen_gf_iadd__m512) {
    size_t num_operations = 0;
    CircuitBuilder builder;
    auto target = builder.append_register(512);
    auto offset = builder.append_register(512);

    benchmark_go([&]() {
        builder.mut.clear();
        gen_gf_iadd(builder, CircuitGenCtx{}, target, offset);
        num_operations = builder.mut.op_types.size();
    })
        .goal_nanos(120)
        .show_rate("ops", num_operations);
}

BENCHMARK(gen_gf_iadd_controlled__m512) {
    size_t num_operations = 0;
    CircuitBuilder builder;
    auto target = builder.append_register(512);
    auto offset = builder.append_register(512);
    auto control = builder.append_register(1);

    benchmark_go([&]() {
        builder.mut.clear();
        gen_gf_iadd(builder, CircuitGenCtx{}, target, offset, control[0]);
        num_operations = builder.mut.op_types.size();
    })
        .goal_nanos(900)
        .show_rate("ops", num_operations);
}
