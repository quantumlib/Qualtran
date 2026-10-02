#ifndef KICKGEN_UTIL_QUBIT_OR_BIT_OR_BOOL_H
#define KICKGEN_UTIL_QUBIT_OR_BIT_OR_BOOL_H

#include <cstdint>
#include <ostream>

#include "bit_or_bool.h"
#include "qubit_or_true.h"

namespace kickmix {

/// A value that could be a qubit identifier, a bit identifier, or a boolean value.
///
/// When this struct is storing a qubit id, it is bit-identical to a QubitId storing that qubit id.
/// When this struct is storing a bit id, it is bit-identical to a BitId storing that bit id.
struct QubitOrBitOrBool {
    uint32_t tagged_id;

    QubitOrBitOrBool(const uint32_t &val) = delete;
    inline QubitOrBitOrBool() : tagged_id(0) {
    }
    inline QubitOrBitOrBool(bool v) : tagged_id(v ? 1 : 0) {
    }
    inline QubitOrBitOrBool(BitId v) : tagged_id(v.tagged_id) {
    }
    inline QubitOrBitOrBool(QubitId v) : tagged_id(v.tagged_id) {
    }
    inline QubitOrBitOrBool(QubitOrTrue v) : tagged_id(v.tagged_id) {
    }
    inline QubitOrBitOrBool(BitOrBool v) : tagged_id(v.tagged_id) {
    }

    constexpr inline QXZTypeTag8 tag() const {
        return (QXZTypeTag8)(tagged_id >> 29);
    }
    constexpr inline uint32_t untagged_id() const {
        return tagged_id & UNTAGGED_MASK;
    }

    inline bool operator==(bool other) const {
        return tagged_id == (other ? 1 : 0);
    }
    inline bool operator==(BitId other) const {
        return tagged_id == other.tagged_id;
    }
    inline bool operator==(QubitId other) const {
        return tagged_id == other.tagged_id;
    }
    inline bool operator==(BitOrBool other) const {
        return tagged_id == other.tagged_id;
    }
    inline bool operator<(QubitOrBitOrBool other) const {
        return tagged_id < other.tagged_id;
    }
    inline bool operator==(QubitOrBitOrBool other) const {
        return tagged_id == other.tagged_id;
    }

    bool is_bool() const {
        return (tagged_id & TAG_MASK) == (uint32_t)QCTypeTag32::BOOL_VAL;
    }
    bool is_false() const {
        return tagged_id == 0;
    }
    bool is_true() const {
        return tagged_id == 1;
    }
    bool is_bit() const {
        return (tagged_id & TAG_MASK) == (uint32_t)QCTypeTag32::BIT_ID;
    }
    bool is_qubit() const {
        return (tagged_id & TAG_MASK) == (uint32_t)QCTypeTag32::QUBIT_ID;
    }

    explicit inline operator QubitId() const {
        return QubitId(tagged_id & UNTAGGED_MASK);
    }
    explicit inline operator BitId() const {
        return BitId(tagged_id & UNTAGGED_MASK);
    }
    explicit inline operator bool() const {
        return (bool)tagged_id;
    }

    std::string str() const;
};
std::ostream &operator<<(std::ostream &out, const QubitOrBitOrBool &v);

}  // namespace kickmix

#endif
