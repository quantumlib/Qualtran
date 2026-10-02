#ifndef KICKMIX_QUBIT_OR_BIT_H
#define KICKMIX_QUBIT_OR_BIT_H

#include "bit_id.h"
#include "qubit_id.h"

namespace kickmix {
/// A value that could be a qubit identifier or a bit identifier.
///
/// When this struct is storing a qubit id, it is bit-identical to a QubitId storing that qubit id.
/// When this struct is storing a bit id, it is bit-identical to a BitId storing that bit id.
struct QubitOrBit {
    uint32_t tagged_id;

    QubitOrBit(const uint32_t &val) = delete;
    constexpr QubitOrBit() : tagged_id(0) {
    }
    QubitOrBit(QubitId q) : tagged_id(q.tagged_id) {
    }
    QubitOrBit(BitId b) : tagged_id(b.tagged_id) {
    }

    bool operator==(const QubitOrBit &other) const {
        return tagged_id == other.tagged_id;
    }
    bool is_bit() const {
        return (tagged_id & TAG_MASK) == (uint32_t)QCTypeTag32::BIT_ID;
    }
    bool is_qubit() const {
        return (tagged_id & TAG_MASK) == (uint32_t)QCTypeTag32::QUBIT_ID;
    }
    constexpr inline uint32_t untagged_id() const {
        return tagged_id & UNTAGGED_MASK;
    }
    explicit inline operator QubitId() const {
        return QubitId(tagged_id & UNTAGGED_MASK);
    }
    explicit inline operator BitId() const {
        return BitId(tagged_id & UNTAGGED_MASK);
    }
};
inline std::ostream &operator<<(std::ostream &out, const QubitOrBit &rhs) {
    if (rhs.is_qubit()) {
        out << (QubitId)rhs;
    } else {
        out << (BitId)rhs;
    }
    return out;
}
}  // namespace kickmix

#endif
