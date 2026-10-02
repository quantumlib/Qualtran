#include "stride_span_z.h"

#include <iostream>
#include <sstream>

#include "array_z.h"

using namespace kickmix;

static const QubitOrBitOrBool GLOBAL_FALSE = false;

stride_span_z stride_span_z::repeat_false(size_t n) {
    return stride_span_z(&GLOBAL_FALSE, 0, n, QXZTypeTag8::BOOL_VAL);
}

std::string stride_span_z::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

std::ostream &kickmix::operator<<(std::ostream &out, const stride_span_z &v) {
    out << "stride_span_z{";
    for (size_t k = 0; k < v.count; k++) {
        if (k) {
            out << ", ";
        }
        out << v[k];
    }
    out << "}";
    return out;
}

stride_span<const QubitId> stride_span_z::checked_cast_to_qubit_ids(const char *value_name_for_error) const {
    if (!is_all_qubits()) {
        std::stringstream ss;
        ss << value_name_for_error;
        ss << " contained values that weren't qubit ids: ";
        ss << *this;
        throw std::invalid_argument(ss.str());
    }
    return cast_data<QubitId>();
}

stride_span<const BitId> stride_span_z::checked_cast_to_bit_ids(const char *value_name_for_error) const {
    if (!is_all_bits()) {
        std::stringstream ss;
        ss << value_name_for_error;
        ss << " contained values that weren't bit ids: ";
        ss << *this;
        throw std::invalid_argument(ss.str());
    }
    return cast_data<BitId>();
}

bool stride_span_z::is_all_qubits() const {
    if (common_type == QXZTypeTag8::QUBIT_ID) {
        return true;
    }
    if (common_type != QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_ZLIKE && count > 0) {
        return false;
    }
    for (auto e : *this) {
        if (!e.is_qubit()) {
            return false;
        }
    }
    return true;
}

bool stride_span_z::is_all_bits() const {
    if (common_type == QXZTypeTag8::BIT_ID) {
        return true;
    }
    if (common_type != QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_ZLIKE && count > 0) {
        return false;
    }
    for (auto e : *this) {
        if (!e.is_bit()) {
            return false;
        }
    }
    return true;
}

bool stride_span_z::is_all_classical() const {
    if (common_type == QXZTypeTag8::BIT_ID || common_type == QXZTypeTag8::BOOL_VAL) {
        return true;
    }
    if (common_type != QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_ZLIKE && count > 0) {
        return false;
    }
    for (auto e : *this) {
        if (!e.is_bit() && !e.is_bool()) {
            return false;
        }
    }
    return true;
}

void stride_span_z::recompute_common_type_data() {
    if (count == 0) {
        common_type = QXZTypeTag8::BOOL_VAL;
        return;
    }
    const uint32_t *v = (const uint32_t *)ptr;
    uint32_t t = *v & TAG_MASK;
    for (size_t k = 1; k < count; k++) {
        v += stride;
        if ((*v & TAG_MASK) != t) {
            common_type = QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_ZLIKE;
            return;
        }
    }
    common_type = (QXZTypeTag8)(t >> 29);
}
