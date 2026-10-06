#include "gen_linear_map.h"

#include "kickmix/util/gf2_field.h"
#include "perf.perf.h"

using namespace kickmix;

BENCHMARK(gen_linear_map__m256) {
    // The Frobenius map of GF(2^256) is a dense, invertible, representative linear map.
    GF2Field field(256);
    GF2Matrix matrix = field.frobenius_matrix(1);

    size_t num_operations = 0;
    CircuitBuilder builder;
    auto target = builder.append_register(256);

    benchmark_go([&]() {
        builder.mut.clear();
        gen_linear_map(builder, CircuitGenCtx{}, target, matrix);
        num_operations = builder.mut.op_types.size();
    })
        .goal_micros(190)
        .show_rate("ops", num_operations);
}

BENCHMARK(gen_linear_map__m64) {
    GF2Field field(64);
    GF2Matrix matrix = field.frobenius_matrix(1);

    size_t num_operations = 0;
    CircuitBuilder builder;
    auto target = builder.append_register(64);

    benchmark_go([&]() {
        builder.mut.clear();
        gen_linear_map(builder, CircuitGenCtx{}, target, matrix);
        num_operations = builder.mut.op_types.size();
    })
        .goal_micros(13)
        .show_rate("ops", num_operations);
}
