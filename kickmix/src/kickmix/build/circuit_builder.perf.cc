#include "circuit_builder.h"

#include <iostream>

#include "perf.perf.h"

using namespace kickmix;

BENCHMARK(builder_c_left_rotate) {
    CircuitBuilder builder;
    QubitId control = builder.append_register(1)[0];
    std::vector<QubitId> target = builder.append_register(256);
    builder.cleft_rotate(control, target);
    size_t n = 0;

    benchmark_go([&]() {
        builder.cleft_rotate(control, target);
        n += builder.mut.op_types.size();
        builder.mut.op_types.clear();
        builder.mut.qqq0.clear();
        builder.mut.qqq1.clear();
        builder.mut.qqq2.clear();
        builder.mut.qq0.clear();
        builder.mut.qq1.clear();
    })
        .goal_nanos(340)
        .show_rate("ops", 255);

    if (n == SIZE_MAX) {
        std::cerr << "data dependence\n";
    }
}

BENCHMARK(builder_pattern256) {
    CircuitBuilder builder;
    // BitId mx{0};
    std::vector<QubitId> sa = builder.reserve_qubits(256);
    std::vector<QubitId> sb = builder.reserve_qubits(256);
    stride_span<const QubitId> a = sa;
    stride_span<const QubitId> b = sb;
    size_t dep = 0;

    benchmark_go([&]() {
        builder.for_each(0, 256 - 1, [&](LoopBuilder &loop, iota k) {
            loop.ccx(a[k], b[k], a[k + 1]);
        });

        dep += builder.mut.op_types.size();
        builder.mut.op_types.clear();
        builder.mut.qqq0.clear();
        builder.mut.qqq1.clear();
        builder.mut.qqq2.clear();
        builder.mut.qq0.clear();
        builder.mut.qq1.clear();
        builder.mut.q0.clear();
        builder.mut.bc.clear();
        builder.mut.b0.clear();
    })
        .goal_nanos(140)
        .show_rate("ops", 255);

    if (dep == SIZE_MAX) {
        std::cerr << "data dependence\n";
    }
}

BENCHMARK(builder_ccx_piecemeal_mixed) {
    CircuitBuilder builder;
    std::vector<QubitOrBitOrBool> a;
    std::vector<QubitOrBitOrBool> b;
    std::vector<QubitOrXBitOrXBool> t;
    for (size_t k = 0; k < 256; k++) {
        if (k % 3 == 0) {
            a.push_back(QubitId(k));
        } else if (k % 3 == 1) {
            a.push_back(BitId(k));
        } else {
            a.push_back((bool)(k % 2));
        }
        if (k / 3 % 3 == 0) {
            b.push_back(QubitId(k));
        } else if (k / 3 % 3 == 1) {
            b.push_back(BitId(k));
        } else {
            b.push_back((bool)(k % 2));
        }
        if (k / 27 % 3 == 0) {
            t.push_back(QubitId(k));
        } else if (k / 27 % 3 == 1) {
            t.push_back(XBitId(k));
        } else {
            t.push_back(k % 2 ? MINUS_KET : PLUS_KET);
        }
    }
    size_t dep = 0;

    benchmark_go([&]() {
        for (size_t k = 0; k < 256; ++k) {
            builder.ccx(a[k], b[k], t[k]);
        }

        dep += builder.mut.op_types.size();
        builder.mut.op_types.clear();
        builder.mut.qqq0.clear();
        builder.mut.qqq1.clear();
        builder.mut.qqq2.clear();
        builder.mut.qq0.clear();
        builder.mut.qq1.clear();
        builder.mut.q0.clear();
        builder.mut.bc.clear();
        builder.mut.b0.clear();
    })
        .goal_nanos(1900)
        .show_rate("ops", 256);

    if (dep == SIZE_MAX) {
        std::cerr << "data dependence\n";
    }
}
