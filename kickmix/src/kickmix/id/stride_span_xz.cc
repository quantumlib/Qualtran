#include "stride_span_xz.h"

#include <iostream>
#include <sstream>

#include "array_xz.h"
#include "array_z.h"

using namespace kickmix;

static const QubitOrXZBitOrXZBool GLOBAL_FALSE = false;

stride_span_xz stride_span_xz::repeat_false(size_t n) {
    return stride_span_xz(&GLOBAL_FALSE, 0, n, QXZTypeTag8::BOOL_VAL);
}

std::string stride_span_xz::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

std::ostream &kickmix::operator<<(std::ostream &out, const stride_span_xz &v) {
    out << "stride_span_xz{";
    for (size_t k = 0; k < v.count; k++) {
        if (k) {
            out << ", ";
        }
        out << v[k];
    }
    out << "}";
    return out;
}

stride_span<const QubitId> stride_span_xz::checked_cast_to_qubit_ids(const char *value_name_for_error) const {
    if (!is_all_qubit_ids()) {
        std::stringstream ss;
        ss << value_name_for_error;
        ss << " contained values that weren't qubit ids: ";
        ss << *this;
        throw std::invalid_argument(ss.str());
    }
    return cast_data<const QubitId>();
}

stride_span<const XBitId> stride_span_xz::checked_cast_to_xbit_ids(const char *value_name_for_error) const {
    if (!is_all_xbit_ids()) {
        std::stringstream ss;
        ss << value_name_for_error;
        ss << " contained values that weren't xbit ids: ";
        ss << *this;
        throw std::invalid_argument(ss.str());
    }
    return cast_data<const XBitId>();
}

stride_span_z stride_span_xz::checked_cast_to_qubit_or_bit_or_bool(const char *value_name_for_error) const {
    if (!is_all_zlike()) {
        std::stringstream ss;
        ss << value_name_for_error;
        ss << " contained values that weren't qubit ids, bit ids, or bools: ";
        ss << *this;
        throw std::invalid_argument(ss.str());
    }
    return stride_span_z((const QubitOrBitOrBool *)ptr, stride, count, common_type);
}
stride_span<const QubitOrBit> stride_span_xz::checked_cast_to_qubit_or_bit(const char *value_name_for_error) const {
    if (!is_all_qubit_or_bit()) {
        std::stringstream ss;
        ss << value_name_for_error;
        ss << " contained values that weren't qubit ids or bit ids: ";
        ss << *this;
        throw std::invalid_argument(ss.str());
    }
    return stride_span<const QubitOrBit>((const QubitOrBit *)ptr, stride, count);
}
stride_span<const QubitOrXBitOrXBool> stride_span_xz::checked_cast_to_qubit_or_xbit_or_xbool(
    const char *value_name_for_error) const {
    if (!is_all_xlike()) {
        std::stringstream ss;
        ss << value_name_for_error;
        ss << " contained values that weren't qubit ids, xbit ids, or xbools: ";
        ss << *this;
        throw std::invalid_argument(ss.str());
    }
    return cast_data<const QubitOrXBitOrXBool>();
}

stride_span<const BitId> stride_span_xz::checked_cast_to_bit_ids(const char *value_name_for_error) const {
    if (!is_all_bit_ids()) {
        std::stringstream ss;
        ss << value_name_for_error;
        ss << " contained values that weren't bit ids: ";
        ss << *this;
        throw std::invalid_argument(ss.str());
    }
    return cast_data<const BitId>();
}

static bool is_all(const stride_span_xz &qc, QCTypeTag32 tag) {
    if (qc.count > 0 && is_not_mixed(qc.common_type)) {
        return (uint32_t)qc.common_type == ((uint32_t)tag >> 29);
    }
    for (size_t k = 0; k < qc.count; k++) {
        if ((qc.ptr[k * qc.stride].tagged_id & TAG_MASK) != (uint32_t)tag) {
            return false;
        }
    }
    return true;
}

bool stride_span_xz::is_all_qubit_ids() const {
    return is_all(*this, QCTypeTag32::QUBIT_ID);
}
bool stride_span_xz::is_all_bit_ids() const {
    return is_all(*this, QCTypeTag32::BIT_ID);
}
bool stride_span_xz::is_all_xbit_ids() const {
    return is_all(*this, QCTypeTag32::XBIT_ID);
}
bool stride_span_xz::is_all_xbools() const {
    return is_all(*this, QCTypeTag32::XBOOL_VAL);
}
bool stride_span_xz::is_all_bools() const {
    return is_all(*this, QCTypeTag32::BOOL_VAL);
}
bool stride_span_xz::is_all_xlike() const {
    switch (common_type) {
        case QXZTypeTag8::XBIT_ID:
        case QXZTypeTag8::XBOOL_VAL:
        case QXZTypeTag8::QUBIT_ID:
        case QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_XLIKE:
            return true;
        case QXZTypeTag8::BIT_ID:
        case QXZTypeTag8::BOOL_VAL:
            if (count > 0) {
                return false;
            }
        default:
            break;
    }
    for (size_t k = 0; k < count; k++) {
        switch ((QCTypeTag32)(ptr[k * stride].tagged_id & TAG_MASK)) {
            case QCTypeTag32::BIT_ID:
            case QCTypeTag32::BOOL_VAL:
                return false;
            default:
                break;
        }
    }
    return true;
}
bool stride_span_xz::is_all_qubit_or_bit() const {
    switch (common_type) {
        case QXZTypeTag8::BIT_ID:
        case QXZTypeTag8::QUBIT_ID:
            return true;
        case QXZTypeTag8::BOOL_VAL:
        case QXZTypeTag8::XBIT_ID:
        case QXZTypeTag8::XBOOL_VAL:
            if (count > 0) {
                return false;
            }
        default:
            break;
    }
    for (size_t k = 0; k < count; k++) {
        switch ((QCTypeTag32)(ptr[k * stride].tagged_id & TAG_MASK)) {
            case QCTypeTag32::XBIT_ID:
            case QCTypeTag32::XBOOL_VAL:
            case QCTypeTag32::BOOL_VAL:
                return false;
            default:
                break;
        }
    }
    return true;
}
bool stride_span_xz::is_all_zlike() const {
    switch (common_type) {
        case QXZTypeTag8::BIT_ID:
        case QXZTypeTag8::BOOL_VAL:
        case QXZTypeTag8::QUBIT_ID:
        case QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_ZLIKE:
            return true;
        case QXZTypeTag8::XBIT_ID:
        case QXZTypeTag8::XBOOL_VAL:
            if (count > 0) {
                return false;
            }
        default:
            break;
    }
    for (size_t k = 0; k < count; k++) {
        switch ((QCTypeTag32)(ptr[k * stride].tagged_id & TAG_MASK)) {
            case QCTypeTag32::XBIT_ID:
            case QCTypeTag32::XBOOL_VAL:
                return false;
            default:
                break;
        }
    }
    return true;
}
bool stride_span_xz::is_all_classical_and_xlike() const {
    switch (common_type) {
        case QXZTypeTag8::XBIT_ID:
        case QXZTypeTag8::XBOOL_VAL:
            return true;
        case QXZTypeTag8::QUBIT_ID:
        case QXZTypeTag8::BIT_ID:
        case QXZTypeTag8::BOOL_VAL:
        case QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_ZLIKE:
            if (count > 0) {
                return false;
            }
        default:
            break;
    }
    for (size_t k = 0; k < count; k++) {
        switch ((QCTypeTag32)(ptr[k * stride].tagged_id & TAG_MASK)) {
            case QCTypeTag32::XBIT_ID:
            case QCTypeTag32::XBOOL_VAL:
                break;
            default:
                return false;
        }
    }
    return true;
}
bool stride_span_xz::is_all_classical_and_zlike() const {
    switch (common_type) {
        case QXZTypeTag8::BIT_ID:
        case QXZTypeTag8::BOOL_VAL:
            return true;
        case QXZTypeTag8::QUBIT_ID:
        case QXZTypeTag8::XBIT_ID:
        case QXZTypeTag8::XBOOL_VAL:
        case QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_XLIKE:
            if (count > 0) {
                return false;
            }
        default:
            break;
    }
    for (size_t k = 0; k < count; k++) {
        switch ((QCTypeTag32)(ptr[k * stride].tagged_id & TAG_MASK)) {
            case QCTypeTag32::BIT_ID:
            case QCTypeTag32::BOOL_VAL:
                break;
            default:
                return false;
        }
    }
    return true;
}

bool stride_span_xz::has_same_contents_as(const stride_span_xz &other) const {
    if (count != other.count) {
        return false;
    }
    for (size_t k = 0; k < count; k++) {
        if (ptr[k * stride] != other.ptr[k * other.stride]) {
            return false;
        }
    }
    return true;
}

QXZTypeTag8 stride_span_xz::compute_common_type() const {
    if (count == 0) {
        return QXZTypeTag8::QUBIT_ID;
    }

    uint32_t t0 = ptr[0].tagged_id & TAG_MASK;
    for (size_t k = 1;; k++) {
        if (k == count) {
            return (QXZTypeTag8)(t0 >> 29);
        }
        if ((ptr[k * stride].tagged_id & TAG_MASK) != t0) {
            break;
        }
    }

    bool allow_x = true;
    bool allow_z = true;
    for (size_t k = 0; k < count; k++) {
        QCTypeTag32 t = (QCTypeTag32)(ptr[k * stride].tagged_id & TAG_MASK);
        allow_z &= t != QCTypeTag32::XBIT_ID && t != QCTypeTag32::XBOOL_VAL;
        allow_x &= t != QCTypeTag32::BIT_ID && t != QCTypeTag32::BOOL_VAL;
    }
    // Note: Either allow_x == false or allow_z == false because there must be at least
    // one non-qubit in the contents (otherwise the earlier loop would have returned QXZTypeTag8::QubitId).
    if (allow_z) {
        return QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_ZLIKE;
    } else if (allow_x) {
        return QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_XLIKE;
    } else {
        return QXZTypeTag8::MAY_BE_MIXED;
    }
}
