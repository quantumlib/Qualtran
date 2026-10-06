#include "main_util.h"

#include <algorithm>
#include <chrono>
#include <cstdint>

#include "kickmix/sim/sim.h"
#include "kickmix/util/mod_rand.h"

using namespace kickmix;

std::string kickmix::big_int_to_str(size_t num_bits, const FixedWidthInt &words, bool decimal, bool hex) {
    std::string result;
    if (hex) {
        for (size_t k = 0; k < num_bits; k += 4) {
            uint8_t b = (words.words[k / 64] >> (k % 64)) & 0xF;
            result.push_back("0123456789ABCDEF"[b]);
        }
        std::reverse(result.data(), result.data() + result.size());
    } else if (decimal) {
        result.append(words.decimal());
    } else {
        for (size_t k = 0; k < num_bits; k++) {
            bool b = (words.words[k / 64] >> (k % 64)) & 1;
            result.push_back("01"[b]);
        }
        std::reverse(result.data(), result.data() + result.size());
    }
    return result;
}
