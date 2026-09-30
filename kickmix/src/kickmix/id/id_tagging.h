#ifndef KICKMIX_ID_TAGGING_H
#define KICKMIX_ID_TAGGING_H

#include <cstdint>
#include <sstream>

namespace kickmix {

// The top three bits of `uint32_t tagged_id` identify whether the value is
// a qubit id, a bit id, a bool, or etc. This allows structs like QubitId, BitId,
// QubitOrBitOrBool, BitOrBool, and etc to all use bit-identical representations
// in memory so that it is safe to alias pointers to one as pointers to another.
// (Though of course the tags at runtime must agree with the compile time types
// in order for this to be well behaved.)
constexpr uint32_t TAG_MASK = 0xE0000000;
constexpr uint32_t UNTAGGED_MASK = 0x1FFFFFFF;

enum class QXZTypeTag8 : uint8_t {
    BOOL_VAL = 0,                         /// False or True. Z-like.
    BIT_ID = 1,                           /// A bit identifier. Z-like.
    QUBIT_ID = 2,                         /// A qubit identifier. Z-like and X-like.
    XBOOL_VAL = 3,                        /// |+> or |->. X-like.
    XBIT_ID = 4,                          /// A bit identifier interpreted as an X basis value. X-like.
    MAY_BE_MIXED_BUT_IS_ALL_XLIKE = 64,   /// Unknown X-like values (QubitId, XBitId, or XBool).
    MAY_BE_MIXED_BUT_IS_ALL_ZLIKE = 128,  /// Unknown Z-like values (QubitId, BitId, or bool).
    MAY_BE_MIXED = 128 | 64,              /// Unknown values.
};
inline constexpr bool is_not_mixed(QXZTypeTag8 tag) {
    return (uint8_t)tag <= (uint8_t)QXZTypeTag8::XBIT_ID;
}
inline std::ostream &operator<<(std::ostream &out, const QXZTypeTag8 &val) {
    if (val == QXZTypeTag8::BOOL_VAL) {
        out << "QXZTypeTag8::BOOL_VAL";
    } else if (val == QXZTypeTag8::XBOOL_VAL) {
        out << "QXZTypeTag8::X_BOOL_VAL";
    } else if (val == QXZTypeTag8::BIT_ID) {
        out << "QXZTypeTag8::BIT_ID";
    } else if (val == QXZTypeTag8::XBIT_ID) {
        out << "QXZTypeTag8::X_BIT_ID";
    } else if (val == QXZTypeTag8::QUBIT_ID) {
        out << "QXZTypeTag8::QUBIT_ID";
    } else if (val == QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_ZLIKE) {
        out << "QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_ZLIKE";
    } else if (val == QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_XLIKE) {
        out << "QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_XLIKE";
    } else if (val == QXZTypeTag8::MAY_BE_MIXED) {
        out << "QXZTypeTag8::MAY_BE_MIXED";
    } else {
        out << "QXZTypeTag8{" << (int)val << "}";
    }
    return out;
}

enum class QCTypeTag32 : uint32_t {
    BOOL_VAL = (uint32_t)QXZTypeTag8::BOOL_VAL << 29,    /// False or True.
    BIT_ID = (uint32_t)QXZTypeTag8::BIT_ID << 29,        /// A bit identifier.
    QUBIT_ID = (uint32_t)QXZTypeTag8::QUBIT_ID << 29,    /// A qubit identifier.
    XBOOL_VAL = (uint32_t)QXZTypeTag8::XBOOL_VAL << 29,  /// |+> or |->
    XBIT_ID = (uint32_t)QXZTypeTag8::XBIT_ID << 29,      /// A bit identifier interpreted as an X basis value.
};

constexpr uint32_t MAX_NUM_QUBITS = 500000000;
constexpr uint32_t MAX_NUM_BITS = 500000000;

}  // namespace kickmix

#endif
