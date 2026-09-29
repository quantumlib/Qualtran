#include "array_z.h"

#include <iostream>
#include <sstream>

using namespace kickmix;

array_z array_z::alloc_noinit(size_t count, QXZTypeTag8 tag) {
    array_z result;
    result.items = (QubitOrBitOrBool *)malloc(count * sizeof(uint32_t));
    result.count = count;
    result.common_type = tag;
    return result;
}

array_z::array_z() {
}
array_z::array_z(array_z &&other) noexcept : items(other.items), count(other.count), common_type(other.common_type) {
    other.items = nullptr;
    other.count = 0;
}
array_z &array_z::operator=(array_z &&other) noexcept {
    if (items != nullptr) {
        free(items);
    }
    items = other.items;
    count = other.count;
    common_type = other.common_type;
    other.items = nullptr;
    other.count = 0;
    return *this;
}

array_z::~array_z() {
    if (items != nullptr) {
        free(items);
    }
}

template <typename T>
array_z concat_helper(stride_span_z lhs, stride_span<const T> rhs) {
    array_z result = array_z::alloc_noinit(lhs.size() + rhs.size());

    QubitOrBitOrBool *tagged_out = result.items;
    for (auto e : lhs) {
        *tagged_out++ = e;
    }
    for (auto e : rhs) {
        *tagged_out++ = e;
    }
    result.recompute_common_type();

    return result;
}

array_z array_z::copy_of_concat(const stride_span_z &lhs, const stride_span_z &rhs) {
    return concat_helper(lhs, rhs.cast_data<QubitOrBitOrBool>());
}

void array_z::recompute_common_type() {
    if (count == 0) {
        common_type = QXZTypeTag8::BOOL_VAL;
        return;
    }
    QubitOrBitOrBool *v = items;
    uint32_t t = v->tagged_id & TAG_MASK;
    for (size_t k = 1; k < count; k++) {
        v += 1;
        if ((v->tagged_id & TAG_MASK) != t) {
            common_type = QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_ZLIKE;
            return;
        }
    }
    common_type = (QXZTypeTag8)(t >> 29);
}

bool array_z::operator==(const array_z &other) const {
    if (count != other.count) {
        return false;
    }
    // DIDNTDO: use memcmp if stride is 1.
    for (size_t k = 0; k < count; k++) {
        if (items[k] != other.items[k]) {
            return false;
        }
    }
    return true;
}

std::string array_z::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

std::ostream &kickmix::operator<<(std::ostream &out, const array_z &v) {
    out << v.as_ptr();
    return out;
}
