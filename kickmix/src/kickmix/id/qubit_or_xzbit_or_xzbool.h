#ifndef KICKGEN_UTIL_QUBIT_OR_XZ_BIT_OR_XZ_BOOL_H
#define KICKGEN_UTIL_QUBIT_OR_XZ_BIT_OR_XZ_BOOL_H

#include <cstdint>

#include "qubit_id.h"
#include "qubit_or_bit_or_bool.h"
#include "qubit_or_minus_state.h"
#include "qubit_or_xbit_or_xbool.h"
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
struct QubitOrXZBitOrXZBool {
    uint32_t tagged_id;

    QubitOrXZBitOrXZBool(const uint32_t &val) = delete;
    constexpr QubitOrXZBitOrXZBool() : tagged_id(0) {
    }
    constexpr QubitOrXZBitOrXZBool(bool bool_val) : tagged_id(bool_val ? 1 : 0) {
    }
    constexpr QubitOrXZBitOrXZBool(XBool xbool_val) : tagged_id(xbool_val.tagged_id) {
    }
    constexpr QubitOrXZBitOrXZBool(XBitId xbit) : tagged_id(xbit.tagged_id) {
    }
    constexpr QubitOrXZBitOrXZBool(BitId bit) : tagged_id(bit.tagged_id) {
    }
    constexpr QubitOrXZBitOrXZBool(QubitId qubit) : tagged_id(qubit.tagged_id) {
    }
    constexpr QubitOrXZBitOrXZBool(QubitOrMinusState qubit_or) : tagged_id(qubit_or.tagged_id) {
    }
    QubitOrXZBitOrXZBool(QubitOrBitOrBool val) : tagged_id(val.tagged_id) {
    }
    QubitOrXZBitOrXZBool(QubitOrXBitOrXBool val) : tagged_id(val.tagged_id) {
    }

    constexpr inline QXZTypeTag8 tag() const {
        return (QXZTypeTag8)(tagged_id >> 29);
    }
    constexpr inline uint32_t untagged_id() const {
        return tagged_id & UNTAGGED_MASK;
    }
    constexpr inline bool operator==(const QubitOrXZBitOrXZBool &other) const {
        return tagged_id == other.tagged_id;
    }
    constexpr inline bool operator<(const QubitOrXZBitOrXZBool &other) const {
        return tagged_id < other.tagged_id;
    }

    constexpr inline bool is_qubit_or_bit_or_bool() const {
        auto tag = tagged_id & TAG_MASK;
        return tag == (uint32_t)QCTypeTag32::QUBIT_ID || tag == (uint32_t)QCTypeTag32::BIT_ID ||
               tag == (uint32_t)QCTypeTag32::BOOL_VAL;
    }
    constexpr inline bool is_qubit_or_xbit_or_xbool() const {
        auto tag = tagged_id & TAG_MASK;
        return tag == (uint32_t)QCTypeTag32::QUBIT_ID || tag == (uint32_t)QCTypeTag32::XBIT_ID ||
               tag == (uint32_t)QCTypeTag32::XBOOL_VAL;
    }
    constexpr inline bool is_xbit() const {
        return (tagged_id & TAG_MASK) == (uint32_t)QCTypeTag32::XBIT_ID;
    }
    constexpr inline bool is_xbool() const {
        return (tagged_id & TAG_MASK) == (uint32_t)QCTypeTag32::XBOOL_VAL;
    }
    constexpr inline bool is_bool() const {
        return (tagged_id & TAG_MASK) == (uint32_t)QCTypeTag32::BOOL_VAL;
    }
    constexpr inline bool is_qubit() const {
        return (tagged_id & TAG_MASK) == (uint32_t)QCTypeTag32::QUBIT_ID;
    }
    constexpr inline bool is_bit() const {
        return (tagged_id & TAG_MASK) == (uint32_t)QCTypeTag32::BIT_ID;
    }
    constexpr inline bool is_plus_ket() const {
        return tagged_id == (uint32_t)QCTypeTag32::XBOOL_VAL;
    }
    constexpr inline bool is_minus_ket() const {
        return tagged_id == ((uint32_t)1 | (uint32_t)QCTypeTag32::XBOOL_VAL);
    }
    constexpr inline bool is_true() const {
        return tagged_id == 1;
    }
    constexpr inline bool is_false() const {
        return tagged_id == 0;
    }

    constexpr explicit inline operator bool() const {
        return is_true();
    }
    constexpr explicit inline operator XBool() const {
        return XBool(is_minus_ket());
    }
    constexpr explicit inline operator BitId() const {
        return BitId(tagged_id & UNTAGGED_MASK);
    }
    constexpr explicit inline operator XBitId() const {
        return XBitId(tagged_id & UNTAGGED_MASK);
    }
    constexpr explicit inline operator QubitId() const {
        return QubitId(tagged_id & UNTAGGED_MASK);
    }
    explicit inline operator QubitOrBitOrBool() const {
        QubitOrBitOrBool result;
        result.tagged_id = tagged_id;
        return result;
    }
    explicit inline operator QubitOrXBitOrXBool() const {
        QubitOrXBitOrXBool result;
        result.tagged_id = tagged_id;
        return result;
    }
    std::string str() const;
};
std::ostream &operator<<(std::ostream &out, const QubitOrXZBitOrXZBool &v);

}  // namespace kickmix

#endif
