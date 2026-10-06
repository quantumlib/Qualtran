#ifndef KICKMIX_XBOOL_H
#define KICKMIX_XBOOL_H

#include <cstdint>
#include <ostream>

#include "id_tagging.h"

namespace kickmix {
/// A boolean that represents whether a value is |-> (true) or |+> (false).
struct XBool {
    uint32_t tagged_id;

    XBool(const uint32_t &val) = delete;
    inline constexpr XBool() : tagged_id((uint32_t)QCTypeTag32::XBOOL_VAL) {
    }
    inline constexpr explicit XBool(bool x_bool_value)
        : tagged_id((x_bool_value ? 1 : 0) | (uint32_t)QCTypeTag32::XBOOL_VAL) {
    }

    inline constexpr bool is_minus_ket() const {
        return tagged_id == (1 | (uint32_t)QCTypeTag32::XBOOL_VAL);
    }
    inline constexpr bool is_plus_ket() const {
        return tagged_id == (uint32_t)QCTypeTag32::XBOOL_VAL;
    }
    inline constexpr bool conjugated_by_h() const {
        return is_minus_ket();
    }

    bool operator==(const XBool &other) const = default;

    std::string str() const;
};
std::ostream &operator<<(std::ostream &out, XBool v);

constexpr XBool MINUS_KET(true);
constexpr XBool PLUS_KET(false);

}  // namespace kickmix

#endif
