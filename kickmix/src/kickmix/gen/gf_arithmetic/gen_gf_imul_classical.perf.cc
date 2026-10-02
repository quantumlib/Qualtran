#include "gen_gf_imul_classical.h"

#include "perf.perf.h"

using namespace kickmix;

BENCHMARK(gen_gf_imul_classical__m256) {
    GF2Field field(256);
    GF2Poly constant = field.mod(GF2Poly::from_str("0xDEADBEEFCAFEF00D1234567890ABCDEF"));

    size_t num_operations = 0;
    CircuitBuilder builder;
    auto target = builder.append_register(256);

    // Most of the cost is building the multiplication matrix and synthesizing it, which is exactly
    // what should be measured.
    benchmark_go([&]() {
        builder.mut.clear();
        gen_gf_imul_classical(builder, CircuitGenCtx{}, field, target, constant);
        num_operations = builder.mut.op_types.size();
    })
        .goal_micros(700)
        .show_rate("ops", num_operations);
}

BENCHMARK(gen_gf_idiv_classical__m256) {
    GF2Field field(256);
    GF2Poly constant = field.mod(GF2Poly::from_str("0xDEADBEEFCAFEF00D1234567890ABCDEF"));

    size_t num_operations = 0;
    CircuitBuilder builder;
    auto target = builder.append_register(256);

    benchmark_go([&]() {
        builder.mut.clear();
        gen_gf_idiv_classical(builder, CircuitGenCtx{}, field, target, constant);
        num_operations = builder.mut.op_types.size();
    })
        .goal_micros(900)
        .show_rate("ops", num_operations);
}
