#ifndef KICKMIX_XBIT_ID_H
#define KICKMIX_XBIT_ID_H

#include <cstdint>
#include <ostream>

#include "bit_id.h"
#include "id_tagging.h"

namespace kickmix {
struct BitId;
/// A bit identifier treated as an X basis value, rather than Z basis.
///
/// For example, an XBit can't be given as the control of a CX gate but it can be given
/// as the target. When given as the target, it acts as an X basis control. Concretely,
/// builder.cx(QubitId(2), XBitId(3)) is equivalent to builder.z_if(QubitId(2), BitId(3)).
struct XBitId {
    uint32_t tagged_id;

    constexpr XBitId() : tagged_id((uint32_t)QCTypeTag32::XBIT_ID) {
    }
    explicit constexpr XBitId(uint32_t untagged_id) : tagged_id(untagged_id | (uint32_t)QCTypeTag32::XBIT_ID) {
        if (untagged_id & TAG_MASK) {
            throw std::invalid_argument("XBitId: untagged_id is tagged");
        }
    }
    inline constexpr bool operator==(const XBitId &other) const = default;
    constexpr inline uint32_t untagged_id() const {
        return tagged_id & UNTAGGED_MASK;
    }
    std::string str() const;

    /// Returns the same bit identifier, but as a Z basis bit rather than X basis.
    inline constexpr BitId conjugated_by_h() const {
        return BitId(untagged_id());
    }

    inline constexpr bool operator<(const XBitId &other) const {
        return tagged_id < other.tagged_id;
    }
};
std::ostream &operator<<(std::ostream &out, XBitId v);
}  // namespace kickmix

#endif
