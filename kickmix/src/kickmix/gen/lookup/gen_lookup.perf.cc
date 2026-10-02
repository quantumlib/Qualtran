#include "gen_lookup.h"

#include <iostream>

#include "perf.perf.h"

using namespace kickmix;

BENCHMARK(gen_lookup) {
    size_t num_operations = 0;
    CircuitBuilder builder;
    std::vector<QubitId> address = builder.append_register(8);
    std::vector<QubitId> output = builder.append_register(30);
    std::vector<BitId> table_bits = builder.append_classical_register(30 << 8);
    auto clean = builder.reserve_qubits(10);
    CircuitGenCtx ctx{.clean_workspace = clean};
    benchmark_go([&]() {
        builder.mut.clear();
        gen_lookup(builder, ctx, table_bits, address, output, 'X', true);
        num_operations = builder.mut.op_types.size();
    })
        .goal_micros(65)
        .show_rate("ops", num_operations);

    if (num_operations == 0) {
        std::cerr << "data dependence\n";
    }
}

BENCHMARK(gen_lookup_uncompute) {
    size_t num_operations = 0;
    CircuitBuilder builder;
    std::vector<QubitId> address = builder.append_register(8);
    std::vector<QubitId> output = builder.append_register(30);
    std::vector<BitId> table_bits = builder.append_classical_register(30 << 8);
    auto clean = builder.reserve_qubits(10);
    CircuitGenCtx ctx{.clean_workspace = clean};
    benchmark_go([&]() {
        builder.mut.clear();
        gen_unlookup(builder, ctx, table_bits, address, output);
        num_operations = builder.mut.op_types.size();
    })
        .goal_micros(43)
        .show_rate("ops", num_operations);

    if (num_operations == 0) {
        std::cerr << "data dependence\n";
    }
}
