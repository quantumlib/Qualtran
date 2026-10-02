#include "gen_iadd_control.h"

#include <iostream>

#include "perf.perf.h"

using namespace kickmix;

BENCHMARK(gen_iadd_control__n256) {
    size_t data = 0;
    size_t num_operations = 0;
    CircuitBuilder builder;
    auto target = builder.append_register(256);
    auto offset = builder.append_register(256);
    QubitId carry = builder.append_register(1)[0];
    QubitId control = builder.append_register(1)[0];
    auto clean = builder.append_register(256);
    CircuitGenCtx ctx{.clean_workspace = clean};

    benchmark_go([&]() {
        builder.mut.clear();
        gen_iadd_control(builder, ctx, target, offset, carry, control);
        data += builder.mut.qqq0.size();
        num_operations = builder.mut.op_types.size();
    })
        .goal_micros(1.6)
        .show_rate("ops", num_operations);

    if (!data) {
        std::cerr << "data dependence\n";
    }
}
