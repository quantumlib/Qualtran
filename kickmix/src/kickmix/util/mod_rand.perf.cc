#include "kickmix/util/mod_rand.h"

#include <iostream>

#include "kickmix/simd/simd.perf.h"
#include "kickmix/util/fixed_width_int.h"
#include "perf.perf.h"

using namespace kickmix;

BENCHMARK(generate_bit_striped_random_values_mod) {
    std::vector<uint64_t> out;
    std::vector<uint64_t> buf;
    out.resize(125);
    buf.resize(125);
    FixedWidthInt f(125, -1);
    f.reset_bit(123);
    f.reset_bit(122);

    std::mt19937_64 rng{0};
    benchmark_go([&]() {
        generate_bit_striped_random_values_mod(rng, out, buf, f);
    })
        .goal_nanos(1600)
        .show_rate("bits", 64 * f.num_bits);

    for (size_t k = 0; k < out.size(); k++) {
        if (out[k] == UINT32_MAX) {
            std::cerr << "data dependence\n";
        }
    }
}
