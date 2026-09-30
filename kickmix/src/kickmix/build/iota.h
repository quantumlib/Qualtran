#ifndef KICKGEN_IOTA_H
#define KICKGEN_IOTA_H

#include <concepts>
#include <cstdint>

namespace kickmix {

struct iota {
    int offset;
    bool negated;

    inline iota operator+(int64_t d) const {
        return {offset + (int)d, negated};
    }
    inline iota operator+(uint64_t d) const {
        return {offset + (int)d, negated};
    }
    inline iota operator+(int32_t d) const {
        return {offset + (int)d, negated};
    }
    inline iota operator+(uint32_t d) const {
        return {offset + (int)d, negated};
    }
    inline iota operator+(int16_t d) const {
        return {offset + (int)d, negated};
    }
    inline iota operator+(uint16_t d) const {
        return {offset + (int)d, negated};
    }
    template <std::integral T>
    inline iota operator+(T d) const {
        return {offset + (int)d, negated};
    }
    iota operator-(int d) const {
        return {offset - d, negated};
    }
    iota operator-() const {
        return {-offset, !negated};
    }
};

}  // namespace kickmix

#endif
