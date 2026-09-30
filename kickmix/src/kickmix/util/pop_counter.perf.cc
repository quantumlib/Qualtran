#include "kickmix/util/pop_counter.h"

#include <iostream>

#include "kickmix/simd/simd.perf.h"
#include "perf.perf.h"

using namespace kickmix;

BENCHMARK_EACH_SIZE_W(pop_counter, {
    PopCounter<W> c{};

    W mask{};
    benchmark_go([&]() {
        mask ^= W::from_u32_broadcast(0x11111111ULL);
        mask.u64_iadd(W::from_u32_broadcast(0xBCDEF012ULL));
        c.masked_increments(mask);
    })
        .goal_nanos(
            (W::BIT_SIZE == 256 && !std::is_same<W, b256_polyfill>()) ? 2.5
            : std::is_same<W, b256_polyfill>()                        ? 2.3
            : std::is_same<W, b128_polyfill>()                        ? 2.1
                                                                      : 1.9)
        .show_rate("bit_increments", c.BATCH_SIZE)
        .show_rate("masked_increments", 1);

    for (size_t k = 0; k < 64; k++) {
        if (c.compute_total(k) == UINT32_MAX) {
            std::cerr << "data dependence\n";
        }
    }
})
