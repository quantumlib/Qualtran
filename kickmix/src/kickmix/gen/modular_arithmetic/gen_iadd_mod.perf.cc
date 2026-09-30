#include "gen_iadd_mod.h"

#include <iostream>

#include "perf.perf.h"

using namespace kickmix;

BENCHMARK(gen_iadd_mod__n256) {
    size_t data = 0;
    size_t num_operations = 0;
    CircuitBuilder builder;
    auto target = builder.append_register(256);
    auto offset = builder.append_register(256);
    array_z modulus = builder.append_classical_register_qcarray_result(256);
    modulus.front() = true;
    modulus.back() = true;
    modulus.recompute_common_type();
    QubitOrTrue control = true;
    double btol = INFINITY;
    auto clean = builder.append_register(256);
    CircuitGenCtx ctx{.clean_workspace = clean};

    benchmark_go([&]() {
        builder.mut.clear();
        gen_iadd_mod(builder, ctx, target, offset, modulus, control, btol);
        data += builder.mut.qqq0.size();
        num_operations = builder.mut.op_types.size();
    })
        .goal_micros(27)
        .show_rate("ops", num_operations);

    if (!data) {
        std::cerr << "data dependence\n";
    }
}

BENCHMARK(gen_iadd_mod_inv__n256) {
    size_t data = 0;
    size_t num_operations = 0;
    CircuitBuilder builder;
    auto target = builder.append_register(256);
    auto offset = builder.append_register(256);
    array_z modulus = builder.append_classical_register_qcarray_result(256);
    modulus.front() = true;
    modulus.back() = true;
    modulus.recompute_common_type();
    QubitOrTrue control = true;
    double btol = INFINITY;
    auto clean = builder.append_register(256);
    CircuitGenCtx ctx{.clean_workspace = clean};

    benchmark_go([&]() {
        builder.mut.clear();
        gen_isub_mod(builder, ctx, target, offset, modulus, control, btol);
        data += builder.mut.qqq0.size();
        num_operations = builder.mut.op_types.size();
    })
        .goal_micros(30)
        .show_rate("ops", num_operations);

    if (!data) {
        std::cerr << "data dependence\n";
    }
}

BENCHMARK(gen_iadd_mod_approx__n256) {
    array_z modulus =
        array_z::copy_of(FixedWidthInt("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F"));

    size_t data = 0;
    size_t num_operations = 0;
    CircuitBuilder builder;
    auto target = builder.append_register(256);
    auto offset = builder.append_register(256);
    QubitOrTrue control = true;
    double btol = INFINITY;
    auto clean = builder.append_register(256);
    CircuitGenCtx ctx{.clean_workspace = clean};

    benchmark_go([&]() {
        builder.mut.clear();
        gen_iadd_mod(builder, ctx, target, offset, modulus, control, btol);
        data += builder.mut.qqq0.size();
        num_operations = builder.mut.op_types.size();
    })
        .goal_micros(24)
        .show_rate("ops", num_operations);

    if (!data) {
        std::cerr << "data dependence\n";
    }
}
