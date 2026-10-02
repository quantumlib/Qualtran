#include "mutable_circuit.h"

#include <fstream>
#include <iostream>

#include "perf.perf.h"

using namespace kickmix;

BENCHMARK(mutable_circuit_append_from_kmx_text) {
    MutableCircuit circuit;
    std::string path = resolve_testdata_file_path("big_circuit.kmx");
    std::stringstream buffer;
    {
        std::ifstream file(path);
        buffer << file.rdbuf();
    }
    std::string content = buffer.str();

    benchmark_go([&]() {
        circuit.clear();
        circuit.append_from_kmx_text(content);
    })
        .goal_millis(6)
        .show_rate("instructions", circuit.op_types.size())
        .show_rate("bytes", content.size());

    if (circuit.op_types.size() == 1) {
        std::cerr << "data dependence\n";
    }
}

BENCHMARK(mutable_circuit_append_from_kmx_file) {
    MutableCircuit circuit;
    std::string path = resolve_testdata_file_path("big_circuit.kmx");
    FILE *f = fopen(path.c_str(), "rb");
    fseek(f, 0, SEEK_END);
    size_t file_size = (size_t)ftell(f);
    benchmark_go([&]() {
        rewind(f);
        circuit.clear();
        circuit.append_from_kmx_file(f);
    })
        .goal_millis(6)
        .show_rate("instructions", circuit.op_types.size())
        .show_rate("bytes", file_size);
    fclose(f);

    if (circuit.op_types.size() == 1) {
        std::cerr << "data dependence\n";
    }
}

BENCHMARK(mutable_circuit_from_kmb_file) {
    std::string path = resolve_testdata_file_path("big_circuit.kmb");
    FILE *f = fopen(path.c_str(), "rb");
    fseek(f, 0, SEEK_END);
    size_t file_size = (size_t)ftell(f);
    Circuit circuit;
    benchmark_go([&]() {
        rewind(f);
        circuit = Circuit::from_kmb_file(f);
    })
        .goal_micros(350)
        .show_rate("instructions", circuit.num_ops)
        .show_rate("bytes", file_size);
    fclose(f);

    if (circuit.num_ops == 1) {
        std::cerr << "data dependence\n";
    }
}
