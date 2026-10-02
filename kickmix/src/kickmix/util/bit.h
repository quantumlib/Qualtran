#ifndef KICKMIX_UTIL_BIT_H
#define KICKMIX_UTIL_BIT_H

#include <cstdint>

namespace kickmix {
inline uint8_t bit_length(uint64_t value) {
    uint8_t result = 0;
    if (value > UINT32_MAX) {
        result += 32;
        value >>= 32;
    }

    if (value > UINT16_MAX) {
        result += 16;
        value >>= 16;
    }

    if (value > UINT8_MAX) {
        result += 8;
        value >>= 8;
    }

    if (value & 0b11110000) {
        result += 4;
        value >>= 4;
    }

    if (value & 0b1100) {
        result += 2;
        value >>= 2;
    }

    if (value & 0b10) {
        result += 1;
        value >>= 1;
    }

    return result + value;
}
};  // namespace kickmix

#endif
