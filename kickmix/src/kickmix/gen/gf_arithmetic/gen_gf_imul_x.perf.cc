#include "gen_gf_imul_x.h"

#include "perf.perf.h"

using namespace kickmix;

BENCHMARK(gen_gf_imul_x__m512_k1) {
    GF2Field field(512);

    size_t num_operations = 0;
    CircuitBuilder builder;
    auto target = builder.append_register(512);

    benchmark_go([&]() {
        builder.mut.clear();
        gen_gf_imul_x(builder, CircuitGenCtx{}, field, target, 1);
        num_operations = builder.mut.op_types.size();
    })
        .goal_micros(6)
        .show_rate("ops", num_operations);
}

BENCHMARK(gen_gf_imul_x__m512_km) {
    // Shifting by a whole degree is what the Karatsuba multiplier does, and is the case the
    // deferred rotation optimizes: it costs one cyclic permutation instead of m of them.
    GF2Field field(512);

    size_t num_operations = 0;
    CircuitBuilder builder;
    auto target = builder.append_register(512);

    benchmark_go([&]() {
        builder.mut.clear();
        gen_gf_imul_x(builder, CircuitGenCtx{}, field, target, 512);
        num_operations = builder.mut.op_types.size();
    })
        .goal_micros(18)
        .show_rate("ops", num_operations);
}
