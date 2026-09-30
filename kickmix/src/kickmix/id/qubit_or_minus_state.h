#ifndef KICKGEN_UTIL_QUBIT_OR_MINUS_STATE_H
#define KICKGEN_UTIL_QUBIT_OR_MINUS_STATE_H

#include <ostream>

#include "qubit_id.h"
#include "xbool.h"

namespace kickmix {

/// A value that could be a qubit identifier, or the value |-> (the -1 eigenstate of the X basis).
///
/// X-basis bits act like controls when specified as the target of a NOT gate.
/// For example, CCX(a, b, MINUS_KET) is equivalent to CZ(a, b).
///
/// When this struct is storing a qubit id, it is bit-identical to a QubitId storing that qubit id.
struct QubitOrMinusState {
    uint32_t tagged_id;

    QubitOrMinusState(const uint32_t &val) = delete;
    constexpr QubitOrMinusState(QubitId qubit) : tagged_id(qubit.tagged_id) {
    }
    consteval QubitOrMinusState(XBool item) : tagged_id(item.tagged_id) {
        if (item.is_plus_ket()) {
            throw std::invalid_argument("QubitOrMinusState(XBool(false))");
        }
    }

    bool is_minus_ket() const {
        return tagged_id == ((uint32_t)1 | (uint32_t)QCTypeTag32::XBOOL_VAL);
    }
    bool is_qubit() const {
        return (tagged_id & TAG_MASK) == (uint32_t)QCTypeTag32::QUBIT_ID;
    }
    bool operator==(const QubitOrMinusState &other) const {
        return tagged_id == other.tagged_id;
    }
    constexpr inline uint32_t untagged_id() const {
        return tagged_id & UNTAGGED_MASK;
    }
    explicit inline operator QubitId() const {
        return QubitId(tagged_id & UNTAGGED_MASK);
    }
    std::string str() const {
        if (is_qubit()) {
            return ((QubitId) * this).str();
        } else if (is_minus_ket()) {
            return "|->";
        } else {
            return "{invalid QubitOrMinusState instance}";
        }
    }
};

}  // namespace kickmix

#endif
