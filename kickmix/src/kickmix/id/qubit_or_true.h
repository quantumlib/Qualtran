#ifndef KICKGEN_UTIL_QUBIT_OR_TRUE_H
#define KICKGEN_UTIL_QUBIT_OR_TRUE_H
#include "qubit_id.h"

namespace kickmix {

/// A value that could be a qubit identifier, the boolean value true.
///
/// When this struct is storing a qubit id, it is bit-identical to a QubitId storing that qubit id.
/// When this struct is storing true, it is bit-identical to a QubitOrBitOrBool storing true.
struct QubitOrTrue {
    uint32_t tagged_id;

    QubitOrTrue(const uint32_t &val) = delete;
    consteval QubitOrTrue(bool val) : tagged_id(1) {
        if (!val) {
            throw std::invalid_argument("QubitOrTrue(false)");
        }
    }
    QubitOrTrue(QubitId qubit) : tagged_id(qubit.tagged_id) {
    }
    bool is_qubit() const {
        return (tagged_id & TAG_MASK) == (uint32_t)QCTypeTag32::QUBIT_ID;
    }
    constexpr inline uint32_t untagged_id() const {
        return tagged_id & UNTAGGED_MASK;
    }
    bool is_true() const {
        return tagged_id == 1;
    }
    explicit inline operator QubitId() const {
        return QubitId(tagged_id & UNTAGGED_MASK);
    }
};

}  // namespace kickmix

#endif
