#ifndef KICKMIX_BIT_ID_H
#define KICKMIX_BIT_ID_H

#include <cstdint>
#include <ostream>

#include "id_tagging.h"

namespace kickmix {
struct XBitId;
/// A bit identifier.
///
/// This struct is bit-identical to a QubitOrBitOrBool storing a bit identifier.
struct BitId {
    uint32_t tagged_id;

    constexpr BitId() : tagged_id((uint32_t)QCTypeTag32::BIT_ID) {
    }
    explicit constexpr BitId(uint32_t untagged_id) : tagged_id(untagged_id | (uint32_t)QCTypeTag32::BIT_ID) {
        if (untagged_id & TAG_MASK) {
            throw std::invalid_argument("BitId: untagged_id is tagged");
        }
    }
    inline bool operator==(const BitId &other) const = default;
    constexpr inline uint32_t untagged_id() const {
        return tagged_id & UNTAGGED_MASK;
    }
    std::string str() const;

    /// Returns the same bit identifier, but as an X basis bit rather than Z basis.
    XBitId conjugated_by_h() const;

    inline bool operator<(const BitId &other) const {
        return tagged_id < other.tagged_id;
    }
};
std::ostream &operator<<(std::ostream &out, BitId v);
}  // namespace kickmix

#endif
