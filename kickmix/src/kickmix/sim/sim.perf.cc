#include "kickmix/sim/sim.h"

#include <iostream>

#include "kickmix/simd/simd.perf.h"
#include "perf.perf.h"

using namespace kickmix;

BENCHMARK(inplace_transpose_64x64) {
    std::array<uint64_t, 64> data{};
    data[0] = 1;

    benchmark_go([&]() {
        inplace_transpose_64x64(&data[0]);
    })
        .goal_nanos(100)
        .show_rate("transpositions", 1);

    if (data[0] == 2) {
        std::cerr << "data dependence\n";
    }
}

BENCHMARK_EACH_SIZE_W(sim_sample, {
    Circuit circuit(R"CIRCUIT(
        CCX q0 q1 q12
        CCX q2 q12 q13
        CCX q3 q13 q14
        CCX q4 q14 q15
        CCX q5 q15 q16

        CCX q16 q6 q7

        CX q16 q6
        HMR q16 b0
        CZ q5 q15 if b0

        CX q15 q5
        HMR q15 b0
        CZ q4 q14 if b0

        CX q14 q4
        HMR q14 b0
        CZ q3 q13 if b0

        CX q13 q3
        HMR q13 b0
        CZ q2 q12 if b0

        CX q12 q2
        HMR q12 b0
        CZ q0 q1 if b0

        CX q0 q1
        X q0
    )CIRCUIT");
    Sim<W, false> sim(std::mt19937_64{0});
    sim.configure_for(circuit);

    benchmark_go([&]() {
        sim.clear_for_shot();
        sim.apply(circuit);
    })
        .goal_nanos(
            std::unordered_map<std::string_view, double>{
                {"b512_avx", 50},
                {"b256_avx", 35},
                {"b128_sse", 39},
                {"b512_polyfill", 220},
                {"b256_polyfill", 75},
                {"b128_polyfill", 55},
                {"b64_polyfill", 43},
            }
                .at(W::NAME))
        .show_rate("instructions", circuit.num_ops * sim.BATCH_SIZE)
        .show_rate("shots", sim.BATCH_SIZE);

    for (const auto &q : sim.state_block) {
        for (size_t r = 0; r < sim.SHOT_WORD_COUNT; r++) {
            if (q.v[r] == UINT32_MAX) {
                std::cerr << "data dependence\n";
            }
        }
    }
})

BENCHMARK_EACH_SIZE_W(sim_sample_with_counting, {
    Circuit circuit(R"CIRCUIT(
        APPEND_TO_REGISTER q0 r0
        APPEND_TO_REGISTER q1 r0
        APPEND_TO_REGISTER q2 r0
        APPEND_TO_REGISTER q3 r0
        APPEND_TO_REGISTER q4 r0
        APPEND_TO_REGISTER q5 r0
        APPEND_TO_REGISTER q6 r0
        APPEND_TO_REGISTER q7 r0

        APPEND_TO_REGISTER q12 r1
        APPEND_TO_REGISTER q13 r1
        APPEND_TO_REGISTER q14 r1
        APPEND_TO_REGISTER q15 r1
        APPEND_TO_REGISTER q16 r1

        CCX q0 q1 q12
        CCX q2 q12 q13
        CCX q3 q13 q14
        CCX q4 q14 q15
        CCX q5 q15 q16

        CCX q16 q6 q7

        CX q16 q6
        HMR q16 b0
        CZ q5 q15 if b0

        CX q15 q5
        HMR q15 b0
        CZ q4 q14 if b0

        CX q14 q4
        HMR q14 b0
        CZ q3 q13 if b0

        CX q13 q3
        HMR q13 b0
        CZ q2 q12 if b0

        CX q12 q2
        HMR q12 b0
        CZ q0 q1 if b0

        CX q0 q1
        X q0
    )CIRCUIT");
    Sim<W, true> sim(std::mt19937_64{0});
    sim.configure_for(circuit);

    benchmark_go([&]() {
        sim.clear_for_shot();
        sim.apply(circuit);
    })
        .goal_nanos(
            std::unordered_map<std::string_view, double>{
                {"b512_avx", 120},
                {"b256_avx", 75},
                {"b128_sse", 75},
                {"b512_polyfill", 1300},
                {"b256_polyfill", 180},
                {"b128_polyfill", 120},
                {"b64_polyfill", 95},
            }
                .at(W::NAME))
        .show_rate("instructions", circuit.num_ops * sim.BATCH_SIZE)
        .show_rate("shots", sim.BATCH_SIZE);

    for (const auto &q : sim.state_block) {
        for (size_t r = 0; r < sim.SHOT_WORD_COUNT; r++) {
            if (q.v[r] == UINT32_MAX) {
                std::cerr << "data dependence\n";
            }
        }
    }
    for (const auto &f : sim.new_op_counters) {
        for (size_t k = 0; k < sim.BATCH_SIZE; k++) {
            if (f.compute_total(k) == UINT32_MAX) {
                std::cerr << "data dependence B\n";
            }
        }
    }
})

BENCHMARK_EACH_SIZE_W(sim_sample_with_populates, {
    Circuit circuit(R"CIRCUIT(
        APPEND_TO_REGISTER q0 r0
        APPEND_TO_REGISTER q1 r0
        APPEND_TO_REGISTER q2 r0
        APPEND_TO_REGISTER q3 r0
        APPEND_TO_REGISTER q4 r0
        APPEND_TO_REGISTER q5 r0
        APPEND_TO_REGISTER q6 r0
        APPEND_TO_REGISTER q7 r0

        APPEND_TO_REGISTER q12 r1
        APPEND_TO_REGISTER q13 r1
        APPEND_TO_REGISTER q14 r1
        APPEND_TO_REGISTER q15 r1
        APPEND_TO_REGISTER q16 r1

        CCX q0 q1 q12
        CCX q2 q12 q13
        CCX q3 q13 q14
        CCX q4 q14 q15
        CCX q5 q15 q16

        CCX q16 q6 q7

        CX q16 q6
        HMR q16 b0
        CZ q5 q15 if b0

        CX q15 q5
        HMR q15 b0
        CZ q4 q14 if b0

        CX q14 q4
        HMR q14 b0
        CZ q3 q13 if b0

        CX q13 q3
        HMR q13 b0
        CZ q2 q12 if b0

        CX q12 q2
        HMR q12 b0
        CZ q0 q1 if b0

        CX q0 q1
        X q0
    )CIRCUIT");
    Sim<W, false> sim(std::mt19937_64{0});
    sim.configure_for(circuit);

    benchmark_go([&]() {
        sim.clear_for_shot();
        for (auto &e : sim.qubit_span()) {
            e.randomize(sim.rng);
        }
        sim.copy_bit_packed_state_into_register_buffer();
        std::swap(sim.register_buffers, sim.register_buffers2);
        sim.apply(circuit);
        sim.copy_bit_packed_state_into_register_buffer();
    })
        .goal_nanos(
            std::unordered_map<std::string_view, double>{
                {"b512_avx", 4800},
                {"b256_avx", 2400},
                {"b128_sse", 1000},
                {"b512_polyfill", 4800},
                {"b256_polyfill", 2400},
                {"b128_polyfill", 1200},
                {"b64_polyfill", 460},
            }
                .at(W::NAME))
        .show_rate("instructions", circuit.num_ops * sim.BATCH_SIZE)
        .show_rate("shots", sim.BATCH_SIZE);

    FixedWidthInt t = sim.register_buffers[0][0];
    t.clear_to_zero();
    for (const auto &f : sim.register_buffers) {
        for (const auto &e : f) {
            t ^= e;
        }
    }
    if (!t) {
        std::cerr << "data dependence\n";
    }
})
