#include "kickmix/util/mod_rand.h"

#include <cstring>
#include <iostream>

#include "kickmix/util/word_ops.h"

using namespace kickmix;

void kickmix::generate_bit_striped_random_values_mod(
    std::mt19937_64 &rng, std::span<uint64_t> out, std::span<uint64_t> buf, const FixedWidthInt &modulus) {
    if (out.size() != modulus.num_bits) {
        throw std::invalid_argument("out.size() != modulus.num_bits");
    }
    if (buf.size() != modulus.num_bits) {
        throw std::invalid_argument("buf.size() != modulus.num_bits");
    }
    if (modulus.num_bits == 0 || !modulus[modulus.num_bits - 1]) {
        throw std::invalid_argument("modulus isn't full capacity");
    }

    memset(out.data(), 0, sizeof(uint64_t) * out.size());
    size_t total = 0;
    while (total < 64) {
        uint64_t is_less_than = 0;
        uint64_t saw_difference = 0;
        for (size_t k = modulus.num_bits; k--;) {
            auto r = rng();
            buf[k] = r;
            if (modulus[k]) {
                is_less_than |= ~r & ~saw_difference;
                saw_difference |= ~r;
            } else {
                saw_difference |= r;
            }
        }

        size_t gained = std::popcount(is_less_than);
        if (gained == 64) {
            memcpy(out.data(), buf.data(), sizeof(uint64_t) * buf.size());
            return;
        }
        for (size_t k = 0; k < modulus.num_bits; k++) {
            out[k] = (out[k] << gained) | extract_bits_u64(buf[k], is_less_than);
        }
        total += gained;
    }
}
