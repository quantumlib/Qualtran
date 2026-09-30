#ifndef KICKMIX_BIT_OR_FALSE_H
#define KICKMIX_BIT_OR_FALSE_H

#include "bit_id.h"

namespace kickmix {

struct BitIdOrFalse {
    uint32_t tagged_id;
    BitIdOrFalse() : tagged_id(0) {
    }
    BitIdOrFalse(BitId id) : tagged_id(id.tagged_id) {
    }
    consteval BitIdOrFalse(bool val) : tagged_id(0) {
        if (val) {
            throw std::invalid_argument("BitIdOrFalse(true)");
        }
    }
    bool is_bit() const {
        return (tagged_id & TAG_MASK) == (uint32_t)QCTypeTag32::BIT_ID;
    }
    BitId bit() const {
        return BitId(tagged_id & UNTAGGED_MASK);
    }
    uint32_t untagged_id() const {
        return tagged_id & UNTAGGED_MASK;
    }
    bool operator==(const BitIdOrFalse &rhs) const = default;
};
std::ostream &operator<<(std::ostream &out, const BitIdOrFalse &op);

}  // namespace kickmix

#endif
