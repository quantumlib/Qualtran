#include "kickmix/util/mod_rand.h"

#include "gtest/gtest.h"

#include "kickmix/sim/sim.h"
#include "kickmix/util/circuit_testing.test.h"
#include "test.test.h"

using namespace kickmix;

TEST(init_mod_random, init_mod_random) {
    auto rng = INDEPENDENT_TEST_RNG();
    FixedWidthInt mod(63);
    mod.randomize(rng);
    mod.set_bit(mod.num_bits - 1);
    std::vector<uint64_t> out;
    std::vector<uint64_t> buf;
    out.resize(64);
    buf.resize(mod.num_bits);
    generate_bit_striped_random_values_mod(rng, std::span{out}.subspan(0, mod.num_bits), buf, mod);
    inplace_transpose_64x64(out.data());
    for (size_t k = 0; k < 64; k++) {
        FixedWidthInt t(mod.num_bits);
        t.words[0] = out[k];
        ASSERT_LT(t, mod);
    }
}

TEST(init_mod_random, init_mod_random_small) {
    auto rng = INDEPENDENT_TEST_RNG();
    FixedWidthInt mod(4);
    mod = 9;
    std::vector<uint64_t> out;
    std::vector<uint64_t> buf;
    out.resize(64);
    buf.resize(mod.num_bits);
    std::array<uint64_t, 9> counts{};
    size_t total_samples = 0;
    for (size_t shot = 0; shot < 10000; shot++) {
        memset(out.data(), 0, sizeof(uint64_t) * 64);
        generate_bit_striped_random_values_mod(rng, std::span{out}.subspan(0, mod.num_bits), buf, mod);
        inplace_transpose_64x64(out.data());
        for (size_t k = 0; k < 64; k++) {
            ASSERT_LT(out[k], 9);
            counts[out[k]] += 1;
        }
        total_samples += 64;
    }

    bool failed = false;
    size_t min_hits = total_samples / 10;
    size_t max_hits = total_samples / 8;
    for (size_t k = 0; k < 9; k++) {
        failed |= counts[k] < min_hits;
        failed |= counts[k] > max_hits;
    }
    if (failed) {
        std::stringstream ss;
        ss << "avg: " << (total_samples / 9) << "\n";
        ss << "min_allowed: " << min_hits << "\n";
        ss << "max_allowed: " << max_hits << "\n";
        for (size_t k = 0; k < 9; k++) {
            ss << k << ": " << counts[k] << "\n";
        }
        ASSERT_TRUE(false) << ss.str();
    }
}
