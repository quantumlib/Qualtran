#include "circuit_pattern.h"

#include <cstdlib>
#include <iostream>

#include "perf.perf.h"

using namespace kickmix;

BENCHMARK(gather_u32_from_pattern1x256) {
    CircuitPattern p;
    std::vector<uint32_t> data;
    for (size_t k = 0; k < 2048; k++) {
        data.push_back(k * k + 3);
    }
    MutableCircuit mut;
    size_t reps = 256;
    uint32_t out = 0;

    benchmark_go([&]() {
        p.pat_bc.push_back(stride_ptr<const uint32_t>{.ptr = &data[2], .stride = 3});
        p.dump_into(mut, 0, 256, false);
        out += mut.bc.size();
    })
        .goal_nanos(90)
        .show_rate("gathered_u32", reps * 1);

    if (out == UINT32_MAX) {
        std::cerr << "data dependence\n";
    }
}

BENCHMARK(gather_u32_from_pattern2x256) {
    CircuitPattern p;
    std::vector<uint32_t> data;
    for (size_t k = 0; k < 2048; k++) {
        data.push_back(k * k + 3);
    }
    MutableCircuit mut;
    size_t reps = 256;
    uint32_t out = 0;

    benchmark_go([&]() {
        p.pat_bc.push_back(stride_ptr<const uint32_t>{.ptr = &data[2], .stride = 3});
        p.pat_bc.push_back(stride_ptr<const uint32_t>{.ptr = &data[2047], .stride = -1});
        p.dump_into(mut, 0, 256, false);
        out += mut.bc.size();
    })
        .goal_nanos(190)
        .show_rate("gathered_u32", reps * 2);

    if (out == UINT32_MAX) {
        std::cerr << "data dependence\n";
    }
}

BENCHMARK(gather_u32_from_pattern3x256) {
    CircuitPattern p;
    std::vector<uint32_t> data;
    for (size_t k = 0; k < 2048; k++) {
        data.push_back(k * k + 3);
    }
    MutableCircuit mut;
    size_t reps = 256;
    uint32_t out = 0;

    benchmark_go([&]() {
        p.pat_bc.push_back(stride_ptr<const uint32_t>{.ptr = &data[0], .stride = 1});
        p.pat_bc.push_back(stride_ptr<const uint32_t>{.ptr = &data[2], .stride = 3});
        p.pat_bc.push_back(stride_ptr<const uint32_t>{.ptr = &data[2047], .stride = -1});
        p.dump_into(mut, 0, 256, false);
        out += mut.bc.size();
    })
        .goal_nanos(230)
        .show_rate("gathered_u32", reps * 3);

    if (out == UINT32_MAX) {
        std::cerr << "data dependence\n";
    }
}

BENCHMARK(gather_u32_from_pattern4x256) {
    CircuitPattern p;
    std::vector<uint32_t> data;
    for (size_t k = 0; k < 2048; k++) {
        data.push_back(k * k + 3);
    }
    MutableCircuit mut;
    size_t reps = 256;
    uint32_t out = 0;

    benchmark_go([&]() {
        p.pat_bc.push_back(stride_ptr<const uint32_t>{.ptr = &data[0], .stride = 1});
        p.pat_bc.push_back(stride_ptr<const uint32_t>{.ptr = &data[2], .stride = 3});
        p.pat_bc.push_back(stride_ptr<const uint32_t>{.ptr = &data[2047], .stride = -1});
        p.pat_bc.push_back(stride_ptr<const uint32_t>{.ptr = &data[1999], .stride = -2});
        p.dump_into(mut, 0, 256, false);
        out += mut.bc.size();
    })
        .goal_nanos(840)
        .show_rate("gathered_u32", reps * 4);

    if (out == UINT32_MAX) {
        std::cerr << "data dependence\n";
    }
}

BENCHMARK(gather_u32_from_pattern7x256) {
    CircuitPattern p;
    std::vector<uint32_t> data;
    for (size_t k = 0; k < 2048; k++) {
        data.push_back(k * k + 3);
    }
    MutableCircuit mut;
    size_t reps = 256;
    uint32_t out = 0;

    benchmark_go([&]() {
        p.pat_bc.push_back({.ptr = &data[0], .stride = 1});
        p.pat_bc.push_back({.ptr = &data[2], .stride = 3});
        p.pat_bc.push_back({.ptr = &data[2047], .stride = -1});
        p.pat_bc.push_back({.ptr = &data[1999], .stride = -2});
        p.pat_bc.push_back({.ptr = &data[6], .stride = 1});
        p.pat_bc.push_back({.ptr = &data[100], .stride = 0});
        p.pat_bc.push_back({.ptr = &data[200], .stride = 2});
        p.dump_into(mut, 0, 256, false);
        out += mut.bc.size();
    })
        .goal_nanos(1300)
        .show_rate("gathered_u32", reps * 7);

    if (out == UINT32_MAX) {
        std::cerr << "data dependence\n";
    }
}
