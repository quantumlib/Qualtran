#include <iostream>

#include "gen_imul2_mod.h"
#include "kickmix/build/circuit_builder.h"
#include "perf.perf.h"

using namespace kickmix;

BENCHMARK(gen_imul2_const) {
    array_z mod_reg =
        array_z::copy_of(FixedWidthInt("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F"));

    size_t data = 0;
    size_t num_operations = 0;
    CircuitBuilder builder;
    std::vector<QubitId> target = builder.append_register(256);
    QubitId control = builder.append_register(1)[0];
    auto clean = builder.reserve_qubits(4);
    auto dirty = builder.reserve_qubits(256);
    CircuitGenCtx ctx{.clean_workspace = clean};
    ctx = ctx.with_more_dirty_qubits(dirty);
    benchmark_go([&]() {
        builder.mut.clear();
        gen_imul2_mod(builder, ctx, target, mod_reg, control, 30);
        data += builder.mut.qqq0.size();
        num_operations = builder.mut.op_types.size();
    })
        .goal_micros(4.6)
        .show_rate("ops", num_operations);

    if (!data) {
        std::cerr << "data dependence\n";
    }
}

BENCHMARK(gen_imul2_bits) {
    CircuitBuilder builder;
    std::vector<QubitOrBitOrBool> mod_bits;
    mod_bits.push_back(true);
    for (auto q : builder.reserve_bits(254)) {
        mod_bits.push_back(q);
    }
    mod_bits.push_back(true);

    size_t data = 0;
    size_t num_operations = 0;
    std::vector<QubitId> target = builder.append_register(256);
    QubitId control = builder.append_register(1)[0];
    auto clean = builder.reserve_qubits(4);
    auto dirty = builder.reserve_qubits(256);
    CircuitGenCtx ctx{.clean_workspace = clean};
    ctx = ctx.with_more_dirty_qubits(dirty);
    benchmark_go([&]() {
        builder.mut.clear();
        gen_imul2_mod(builder, ctx, target, mod_bits, control, 30);
        data += builder.mut.qqq0.size();
        num_operations = builder.mut.op_types.size();
    })
        .goal_micros(25)
        .show_rate("ops", num_operations);

    if (!data) {
        std::cerr << "data dependence\n";
    }
}
