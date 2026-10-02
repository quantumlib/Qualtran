#include "gen_gf_mul.h"

#include "perf.perf.h"

using namespace kickmix;

BENCHMARK(gen_gf2_poly_mul__n256) {
    size_t num_operations = 0;
    CircuitBuilder builder;
    auto target = builder.append_register(2 * 256 - 1);
    auto lhs = builder.append_register(256);
    auto rhs = builder.append_register(256);

    benchmark_go([&]() {
        builder.mut.clear();
        gen_gf2_poly_mul(builder, CircuitGenCtx{}, target, lhs, rhs);
        num_operations = builder.mut.op_types.size();
    })
        .goal_micros(850)
        .show_rate("ops", num_operations);
}

BENCHMARK(gen_gf_mul__m256) {
    GF2Field field(256);

    size_t num_operations = 0;
    CircuitBuilder builder;
    auto target = builder.append_register(256);
    auto lhs = builder.append_register(256);
    auto rhs = builder.append_register(256);
    CircuitGenCtx ctx{};

    benchmark_go([&]() {
        builder.mut.clear();
        gen_gf_mul(builder, ctx, field, target, lhs, rhs);
        num_operations = builder.mut.op_types.size();
    })
        .goal_millis(1.6)
        .show_rate("ops", num_operations);
}

BENCHMARK(gen_gf_unmul__m256) {
    // Uncomputing a product is measurement based, so it should be dramatically cheaper to build
    // and dramatically cheaper to run than the product itself.
    GF2Field field(256);

    size_t num_operations = 0;
    CircuitBuilder builder;
    auto target = builder.append_register(256);
    auto lhs = builder.append_register(256);
    auto rhs = builder.append_register(256);
    CircuitGenCtx ctx{};

    benchmark_go([&]() {
        builder.mut.clear();
        gen_gf_unmul(builder, ctx, field, target, lhs, rhs);
        num_operations = builder.mut.op_types.size();
    })
        .goal_micros(1100)
        .show_rate("ops", num_operations);
}
