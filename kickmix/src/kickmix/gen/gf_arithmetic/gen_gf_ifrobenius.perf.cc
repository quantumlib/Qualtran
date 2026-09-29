#include "gen_gf_ifrobenius.h"

#include "perf.perf.h"

using namespace kickmix;

BENCHMARK(gen_gf_ifrobenius__m256_k1) {
    GF2Field field(256);

    size_t num_operations = 0;
    CircuitBuilder builder;
    auto target = builder.append_register(256);

    benchmark_go([&]() {
        builder.mut.clear();
        gen_gf_ifrobenius(builder, CircuitGenCtx{}, field, target, 1);
        num_operations = builder.mut.op_types.size();
    })
        .goal_micros(400)
        .show_rate("ops", num_operations);
}

BENCHMARK(gen_gf_ifrobenius__m256_k128) {
    // A large power is where the original 64 bit implementation overflowed. Building the matrix
    // costs k squarings plus m multiplies, so the cost grows only mildly with k.
    GF2Field field(256);

    size_t num_operations = 0;
    CircuitBuilder builder;
    auto target = builder.append_register(256);

    benchmark_go([&]() {
        builder.mut.clear();
        gen_gf_ifrobenius(builder, CircuitGenCtx{}, field, target, 128);
        num_operations = builder.mut.op_types.size();
    })
        .goal_micros(2300)
        .show_rate("ops", num_operations);
}
