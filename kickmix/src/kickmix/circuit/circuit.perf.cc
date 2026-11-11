#include "circuit.h"

#include <cstdlib>
#include <iostream>

#include "mutable_circuit.h"
#include "perf.perf.h"

using namespace kickmix;

BENCHMARK(circuit_max_push_condition_depth) {
    MutableCircuit m;
    for (size_t k = 0; k < 1001; k++) {
        m.append_from_kmx_text(R"CIRCUIT(
            PUSH_CONDITION if b0
            POP_CONDITION
            X q0
            PUSH_CONDITION if b1
        )CIRCUIT");
    }
    Circuit c = m.to_validated_circuit();

    size_t total = 0;
    benchmark_go([&]() {
        total += c.compute_max_condition_depth();
    })
        .goal_nanos(1700)
        .show_rate("ops", c.num_ops);

    if (total == UINT64_MAX) {
        std::cerr << "data dependence\n";
    }
}

BENCHMARK(circuit_from_kmx) {
    auto path = resolve_testdata_file_path("big_circuit.kmx");
    FILE *f = fopen(path.c_str(), "rb");

    size_t total = 0;
    size_t sum = 0;
    benchmark_go([&]() {
        fseek(f, 0, SEEK_SET);
        Circuit c = Circuit::from_kmx_file(f);
        total = c.num_ops;
        sum += c.num_ops;
    })
        .goal_millis(16)
        .show_rate("ops", total);

    if (sum == UINT64_MAX) {
        std::cerr << "data dependence\n";
    }
}

BENCHMARK(circuit_from_kmb) {
    auto path = resolve_testdata_file_path("big_circuit.kmb");
    FILE *f = fopen(path.c_str(), "rb");

    size_t total = 0;
    size_t sum = 0;
    benchmark_go([&]() {
        fseek(f, 0, SEEK_SET);
        Circuit c = Circuit::from_kmb_file(f);
        total = c.num_ops;
        sum += c.num_ops;
    })
        .goal_micros(640)
        .show_rate("ops", total);

    if (sum == UINT64_MAX) {
        std::cerr << "data dependence\n";
    }
}
