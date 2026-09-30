#include "gen_iadd_classical.h"

#include <iostream>

#include "perf.perf.h"

using namespace kickmix;

BENCHMARK(gen_ixor_carries_from_addition__n256_writekmb) {
    array_z mod_reg =
        array_z::copy_of(FixedWidthInt("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F"));
    FILE *f = tmpfile();

    size_t data = 0;
    size_t num_operations = 0;
    benchmark_go([&]() {
        CircuitBuilder builder;
        std::vector<QubitId> target = builder.append_register(255);
        std::vector<QubitId> dst = builder.append_register(255);
        QubitId carry_in = builder.append_register(1)[0];
        QubitId control = builder.append_register(1)[0];
        auto clean = builder.reserve_qubits(2);
        auto dirty = builder.reserve_qubits(256);
        CircuitGenCtx ctx{.clean_workspace = clean};
        ctx = ctx.with_more_dirty_qubits(dirty);
        ctx.minimize_qubits = false;
        gen_ixor_carries_from_addition(builder, ctx, target, mod_reg, dst, carry_in, control);
        fseek(f, 0, SEEK_SET);
        builder.finish_circuit().write_kmb_to(f);
        data += builder.mut.qqq0.size();
        num_operations = builder.mut.op_types.size();
    })
        .goal_micros(22)
        .show_rate("ops", num_operations);

    fclose(f);
    if (!data) {
        std::cerr << "data dependence\n";
    }
}

BENCHMARK(gen_ixor_carries_from_addition__n256_writekmb_no_validation) {
    array_z mod_reg =
        array_z::copy_of(FixedWidthInt("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F"));
    FILE *f = tmpfile();

    size_t data = 0;
    size_t num_operations = 0;
    benchmark_go([&]() {
        CircuitBuilder builder;
        builder.skip_validation = true;
        std::vector<QubitId> target = builder.append_register(255);
        std::vector<QubitId> dst = builder.append_register(255);
        QubitId carry_in = builder.append_register(1)[0];
        QubitId control = builder.append_register(1)[0];
        auto clean = builder.reserve_qubits(2);
        auto dirty = builder.reserve_qubits(256);
        CircuitGenCtx ctx{.clean_workspace = clean};
        ctx = ctx.with_more_dirty_qubits(dirty);
        ctx.minimize_qubits = false;
        gen_ixor_carries_from_addition(builder, ctx, target, mod_reg, dst, carry_in, control);
        fseek(f, 0, SEEK_SET);
        builder.finish_circuit().write_kmb_to(f);
        data += builder.mut.qqq0.size();
        num_operations = builder.mut.op_types.size();
    })
        .goal_micros(20)
        .show_rate("ops", num_operations);

    fclose(f);
    if (!data) {
        std::cerr << "data dependence\n";
    }
}

BENCHMARK(gen_ixor_carries_from_addition__n256_bits) {
    array_z mod_reg = array_z::alloc_noinit(256, QXZTypeTag8::BIT_ID);
    for (size_t k = 0; k < 256; k++) {
        mod_reg[k] = BitId{(uint32_t)k};
    }

    size_t data = 0;
    size_t num_operations = 0;
    CircuitBuilder builder;
    std::vector<QubitId> target = builder.append_register(255);
    std::vector<QubitId> dst = builder.append_register(255);
    QubitId carry_in = builder.append_register(1)[0];
    QubitId control = builder.append_register(1)[0];
    auto clean = builder.reserve_qubits(2);
    auto dirty = builder.reserve_qubits(256);
    CircuitGenCtx ctx{.clean_workspace = clean};
    ctx = ctx.with_more_dirty_qubits(dirty);
    ctx.minimize_qubits = false;
    benchmark_go([&]() {
        builder.mut.clear();
        gen_ixor_carries_from_addition(builder, ctx, target, mod_reg, dst, carry_in, control);
        data += builder.mut.qqq0.size();
        num_operations = builder.mut.op_types.size();
    })
        .goal_micros(1.1)
        .show_rate("ops", num_operations);

    if (!data) {
        std::cerr << "data dependence\n";
    }
}

BENCHMARK(gen_ixor_carries_from_addition__n256) {
    array_z mod_reg =
        array_z::copy_of(FixedWidthInt("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F"));

    size_t data = 0;
    size_t num_operations = 0;
    CircuitBuilder builder;
    std::vector<QubitId> target = builder.append_register(255);
    std::vector<QubitId> dst = builder.append_register(255);
    QubitId carry_in = builder.append_register(1)[0];
    QubitId control = builder.append_register(1)[0];
    auto clean = builder.reserve_qubits(2);
    auto dirty = builder.reserve_qubits(256);
    CircuitGenCtx ctx{.clean_workspace = clean};
    ctx = ctx.with_more_dirty_qubits(dirty);
    ctx.minimize_qubits = false;
    benchmark_go([&]() {
        builder.mut.clear();
        gen_ixor_carries_from_addition(builder, ctx, target, mod_reg, dst, carry_in, control);
        data += builder.mut.qqq0.size();
        num_operations = builder.mut.op_types.size();
    })
        .goal_micros(1.1)
        .show_rate("ops", num_operations);

    if (!data) {
        std::cerr << "data dependence\n";
    }
}

BENCHMARK(gen_iadd_classical__n256) {
    array_z offset =
        array_z::copy_of(FixedWidthInt("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F"));

    size_t data = 0;
    size_t num_operations = 0;
    CircuitBuilder builder;
    std::vector<QubitId> target = builder.append_register(255);
    std::vector<QubitId> dst = builder.append_register(255);
    QubitId carry_in = builder.append_register(1)[0];
    QubitId control = builder.append_register(1)[0];
    auto clean = builder.reserve_qubits(2);
    auto dirty = builder.reserve_qubits(256);
    CircuitGenCtx ctx{.clean_workspace = clean};
    ctx = ctx.with_more_dirty_qubits(dirty);
    ctx.minimize_qubits = false;
    benchmark_go([&]() {
        builder.mut.clear();
        gen_iadd_classical(builder, ctx, target, offset, carry_in, control, INFINITY);
        data += builder.mut.qqq0.size();
        num_operations = builder.mut.op_types.size();
    })
        .goal_micros(7.5)
        .show_rate("ops", num_operations);

    if (!data) {
        std::cerr << "data dependence\n";
    }
}

BENCHMARK(gen_iadd_classical__n256_partial_clean) {
    array_z offset =
        array_z::copy_of(FixedWidthInt("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F"));

    size_t data = 0;
    size_t num_operations = 0;
    CircuitBuilder builder;
    std::vector<QubitId> target = builder.append_register(255);
    std::vector<QubitId> dst = builder.append_register(255);
    QubitId carry_in = builder.append_register(1)[0];
    QubitId control = builder.append_register(1)[0];
    auto clean = builder.reserve_qubits(100);
    auto dirty = builder.reserve_qubits(256);
    CircuitGenCtx ctx{.clean_workspace = clean};
    ctx = ctx.with_more_dirty_qubits(dirty);
    ctx.minimize_qubits = false;
    benchmark_go([&]() {
        builder.mut.clear();
        gen_iadd_classical(builder, ctx, target, offset, carry_in, control, INFINITY);
        data += builder.mut.qqq0.size();
        num_operations = builder.mut.op_types.size();
    })
        .goal_micros(6.9)
        .show_rate("ops", num_operations);

    if (!data) {
        std::cerr << "data dependence\n";
    }
}
