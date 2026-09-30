#include "kickmix/util/pop_counter.h"

#include "gtest/gtest.h"

#include "kickmix/simd/simd.test.h"
#include "test.test.h"

using namespace kickmix;

TEST_EACH_SIZE_W(pop_counter, all_on, {
    PopCounter<W> counter;
    for (size_t k = 0; k < 500; k++) {
        counter.masked_increments(~W{});
        ASSERT_EQ(counter.compute_total(1), k + 1) << k;
    }
    for (size_t k = 0; k < counter.BATCH_SIZE; k++) {
        EXPECT_EQ(counter.compute_total(k), 500) << k;
    }
})

TEST_EACH_SIZE_W(pop_counter, all_off, {
    PopCounter<W> counter;
    for (size_t k = 0; k < 500; k++) {
        counter.masked_increments(W{});
    }
    for (size_t k = 0; k < counter.BATCH_SIZE; k++) {
        EXPECT_EQ(counter.compute_total(k), 0) << k;
    }
})

TEST_EACH_SIZE_W(pop_counter, fuzz, {
    std::mt19937_64 rng = INDEPENDENT_TEST_RNG();
    PopCounter<W> counter;
    uint64_t counts[counter.BATCH_SIZE]{};
    for (size_t k = 0; k < 10000; k++) {
        auto m = W::random(rng);
        counter.masked_increments(m);
        for (size_t b = 0; b < counter.BATCH_SIZE; b++) {
            if (m.bit(b) & 1) {
                counts[b] += 1;
            }
        }
    }
    for (size_t k = 0; k < counter.BATCH_SIZE; k++) {
        EXPECT_EQ(counter.compute_total(k), counts[k]) << k;
    }
})
