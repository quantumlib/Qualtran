#ifndef KICKGEN_UTIL_QUBIT_OR_X_EIGEN_STATE_H
#define KICKGEN_UTIL_QUBIT_OR_X_EIGEN_STATE_H

#include <cstdint>

#include "qubit_id.h"
#include "qubit_or_minus_state.h"
#include "xbit_id.h"

namespace kickmix {

/// A value that could be a qubit identifier, an X-basis bit identifier, or an X-basis boolean value.
///
/// X-basis bits act like controls when specified as the target of a NOT gate.
/// For example, a CCX(a, b, c) where c is an X basis bit is equivalent to
/// CCZ(a, b, c.bit_value) and CCX(a, b, c) where c is MINUS_KET is equivalent
/// to CZ(a, b).
///
/// When this struct is storing a qubit id, it is bit-identical to a QubitId storing that qubit id.
struct QubitOrXBitOrXBool {
    uint32_t tagged_id;

    QubitOrXBitOrXBool(const uint32_t &val) = delete;
    constexpr QubitOrXBitOrXBool() : tagged_id(XBool(false).tagged_id) {
    }
    constexpr QubitOrXBitOrXBool(XBool value) : tagged_id(value.tagged_id) {
    }
    constexpr QubitOrXBitOrXBool(XBitId xbit) : tagged_id(xbit.tagged_id) {
    }
    constexpr QubitOrXBitOrXBool(QubitId qubit) : tagged_id(qubit.tagged_id) {
    }
    constexpr QubitOrXBitOrXBool(QubitOrMinusState qubit_or) : tagged_id(qubit_or.tagged_id) {
    }

    constexpr inline QXZTypeTag8 tag() const {
        return (QXZTypeTag8)(tagged_id >> 29);
    }
    constexpr inline uint32_t untagged_id() const {
        return tagged_id & UNTAGGED_MASK;
    }
    constexpr inline bool operator==(const QubitId &other) const {
        return tagged_id == other.tagged_id;
    }
    constexpr inline bool operator==(const QubitOrXBitOrXBool &other) const {
        return tagged_id == other.tagged_id;
    }

    constexpr inline bool is_xbit() const {
        return (tagged_id & TAG_MASK) == (uint32_t)QCTypeTag32::XBIT_ID;
    }
    constexpr inline bool is_xbool() const {
        return (tagged_id & TAG_MASK) == (uint32_t)QCTypeTag32::XBOOL_VAL;
    }
    constexpr inline bool is_qubit() const {
        return (tagged_id & TAG_MASK) == (uint32_t)QCTypeTag32::QUBIT_ID;
    }
    constexpr inline bool is_plus_ket() const {
        return tagged_id == (uint32_t)QCTypeTag32::XBOOL_VAL;
    }
    constexpr inline bool is_minus_ket() const {
        return tagged_id == ((uint32_t)1 | (uint32_t)QCTypeTag32::XBOOL_VAL);
    }

    constexpr explicit inline operator XBitId() const {
        return XBitId(tagged_id & UNTAGGED_MASK);
    }
    constexpr explicit inline operator QubitId() const {
        return QubitId(tagged_id & UNTAGGED_MASK);
    }
    std::string str() const;
};
std::ostream &operator<<(std::ostream &out, const QubitOrXBitOrXBool &v);

}  // namespace kickmix

#endif
