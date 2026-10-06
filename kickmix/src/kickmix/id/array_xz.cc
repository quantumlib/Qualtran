#include "array_xz.h"

#include <iostream>
#include <sstream>

using namespace kickmix;

array_xz array_xz::alloc_noinit(size_t count, QXZTypeTag8 tag) {
    array_xz result;
    result.items = (QubitOrXZBitOrXZBool *)malloc(count * sizeof(uint32_t));
    result.count = count;
    result.common_type = tag;
    return result;
}

array_xz::array_xz() {
}
array_xz::array_xz(array_xz &&other) noexcept : items(other.items), count(other.count), common_type(other.common_type) {
    other.items = nullptr;
    other.count = 0;
}
array_xz::array_xz(array_z &&other) noexcept
    : items((QubitOrXZBitOrXZBool *)other.items), count(other.count), common_type(other.common_type) {
    other.items = nullptr;
    other.count = 0;
}
array_xz &array_xz::operator=(array_xz &&other) noexcept {
    if (this == &other) {
        return *this;
    }
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

array_xz::~array_xz() {
    if (items != nullptr) {
        free(items);
    }
}

void array_xz::recompute_common_type() {
    common_type = as_ptr().compute_common_type();
}

bool array_xz::operator==(const array_xz &other) const {
    return as_ptr().has_same_contents_as(other.as_ptr());
}

std::string array_xz::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

std::ostream &kickmix::operator<<(std::ostream &out, const array_xz &v) {
    out << "array_xz{";
    for (size_t k = 0; k < v.count; k++) {
        if (k) {
            out << ", ";
        }
        out << v.as_ptr()[k];
    }
    out << "}";
    return out;
}

array_xz array_xz::copy_of_concat(const stride_span_xz &lhs, const stride_span_xz &rhs) {
    array_xz result;
    size_t n = lhs.size() + rhs.size();
    result.items = (QubitOrXZBitOrXZBool *)malloc(n * sizeof(QubitOrXZBitOrXZBool));
    result.count = n;

    // Copy data.
    QubitOrXZBitOrXZBool *out = result.items;
    for (auto e : lhs) {
        *out++ = e;
    }
    for (auto e : rhs) {
        *out++ = e;
    }
    result.recompute_common_type();

    return result;
}
