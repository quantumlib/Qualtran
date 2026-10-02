#ifndef KICKMIX_QUBIT_ID_H
#define KICKMIX_QUBIT_ID_H

#include <cstdint>
#include <ostream>

#include "id_tagging.h"

namespace kickmix {

/// A qubit identifier.
///
/// This struct is bit-identical to a QubitOrBitOrBool storing a qubit identifier.
struct QubitId {
    uint32_t tagged_id;

    constexpr QubitId() : tagged_id((uint32_t)QCTypeTag32::QUBIT_ID) {
    }
    explicit constexpr QubitId(uint32_t untagged_id) : tagged_id(untagged_id | (uint32_t)QCTypeTag32::QUBIT_ID) {
        if (untagged_id & TAG_MASK) {
            throw std::invalid_argument("QubitId: untagged_id is tagged");
        }
    }
    inline bool operator==(const QubitId &other) const = default;
    constexpr inline uint32_t untagged_id() const {
        return tagged_id & UNTAGGED_MASK;
    }
    std::string str() const;

    inline bool operator<(const QubitId &other) const {
        return tagged_id < other.tagged_id;
    }
};
std::ostream &operator<<(std::ostream &out, QubitId v);

}  // namespace kickmix

#endif
