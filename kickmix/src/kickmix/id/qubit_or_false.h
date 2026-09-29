#ifndef KICKMIX_QUBIT_OR_FALSE_H
#define KICKMIX_QUBIT_OR_FALSE_H

#include "qubit_id.h"

namespace kickmix {

struct QubitOrFalse {
    uint32_t tagged_id;
    QubitOrFalse(const uint32_t &val) = delete;
    QubitOrFalse() : tagged_id(0) {
    }
    QubitOrFalse(QubitId id) : tagged_id(id.tagged_id) {
    }
    bool is_qubit() const {
        return (tagged_id & TAG_MASK) == (uint32_t)QCTypeTag32::QUBIT_ID;
    }
    QubitId qubit() const {
        return QubitId(tagged_id & UNTAGGED_MASK);
    }
    uint32_t untagged_id() const {
        return tagged_id & UNTAGGED_MASK;
    }
    bool operator==(const QubitOrFalse &rhs) const = default;
};
std::ostream &operator<<(std::ostream &out, const QubitOrFalse &op);

}  // namespace kickmix

#endif
