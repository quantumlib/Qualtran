#ifndef KICKGEN_UTIL_BIT_OR_BOOL_H
#define KICKGEN_UTIL_BIT_OR_BOOL_H

#include <cstdint>
#include <cstring>
#include <ostream>
#include <sstream>

#include "bit_id.h"

namespace kickmix {

struct BitOrBool;
std::ostream &operator<<(std::ostream &out, const BitOrBool &v);
struct BitOrBool {
    uint32_t tagged_id;

    BitOrBool(const uint32_t &val) = delete;
    inline BitOrBool() : tagged_id(0) {
    }
    inline BitOrBool(bool v) : tagged_id(v ? 1 : 0) {
    }
    inline BitOrBool(BitId v) : tagged_id(v.tagged_id) {
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
    inline bool operator==(BitOrBool other) const {
        return tagged_id == other.tagged_id;
    }
    inline bool is_bool() const {
        return (tagged_id & TAG_MASK) == (uint32_t)QCTypeTag32::BOOL_VAL;
    }
    inline bool is_bit() const {
        return (tagged_id & TAG_MASK) == (uint32_t)QCTypeTag32::BIT_ID;
    }
    inline explicit operator BitId() const {
        return BitId(tagged_id & UNTAGGED_MASK);
    }
    inline explicit operator bool() const {
        return (bool)tagged_id;
    }
    std::string str() const {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }
};
inline std::ostream &operator<<(std::ostream &out, const BitOrBool &v) {
    if (v.is_bit()) {
        out << (BitId)v;
    } else if ((bool)v) {
        out << "true";
    } else {
        out << "false";
    }
    return out;
}

}  // namespace kickmix

#endif
