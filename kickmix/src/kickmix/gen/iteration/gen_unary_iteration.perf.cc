#include "gen_unary_iteration.h"

#include <iostream>

#include "perf.perf.h"

using namespace kickmix;

BENCHMARK(gen_unary_iteration_sweep) {
    size_t num_operations = 0;
    CircuitBuilder builder;
    std::vector<QubitId> address = builder.append_register(10);
    std::vector<QubitId> ctrl = builder.append_register(1);
    auto clean = builder.reserve_qubits(10);
    CircuitGenCtx ctx{.clean_workspace = clean};
    benchmark_go([&]() {
        builder.mut.clear();
        UnaryIterationCursor cursor(builder, ctx, address, ctrl[0]);
        for (uint64_t v = 0; v < 1024; v++) {
            cursor.move_to(v);
        }
        cursor.close();
        num_operations = builder.mut.op_types.size();
    })
        .goal_micros(99)
        .show_rate("ops", num_operations);

    if (num_operations == 0) {
        std::cerr << "data dependence\n";
    }
}
