#ifndef KICKGEN_PYBIND_CONVERTED_ARRAY_H
#define KICKGEN_PYBIND_CONVERTED_ARRAY_H

#include <pybind11/pybind11.h>

#include "kickmix/id/stride_span_xz.h"

namespace kickmix_py {

struct ConvertedArrayXZ {
    kickmix::stride_span_xz span;
    kickmix::QubitOrXZBitOrXZBool *free_on_destruct;
    bool was_singleton;

    ConvertedArrayXZ() : span(), free_on_destruct(nullptr), was_singleton(false) {
    }
    ConvertedArrayXZ(const kickmix::stride_span_z &span) : span(span), free_on_destruct(nullptr), was_singleton(false) {
    }
    ConvertedArrayXZ(
        const kickmix::stride_span_z &span_z, kickmix::QubitOrXZBitOrXZBool *free_on_destruct, bool was_singleton)
        : span(span_z), free_on_destruct(free_on_destruct), was_singleton(was_singleton) {
    }
    ConvertedArrayXZ(
        const kickmix::stride_span_xz &span_xz, kickmix::QubitOrXZBitOrXZBool *free_on_destruct, bool was_singleton)
        : span(span_xz), free_on_destruct(free_on_destruct), was_singleton(was_singleton) {
    }
    ConvertedArrayXZ(const ConvertedArrayXZ &) = delete;
    ConvertedArrayXZ &operator=(const ConvertedArrayXZ &) = delete;
    ConvertedArrayXZ &operator=(ConvertedArrayXZ &&other) noexcept {
        if (free_on_destruct != nullptr) {
            free(free_on_destruct);
        }
        span = other.span;
        free_on_destruct = other.free_on_destruct;
        was_singleton = other.was_singleton;
        other.free_on_destruct = nullptr;
        return *this;
    }
    ConvertedArrayXZ(ConvertedArrayXZ &&other) noexcept
        : span(other.span), free_on_destruct(other.free_on_destruct), was_singleton(other.was_singleton) {
        other.free_on_destruct = nullptr;
    }
    operator kickmix::stride_span_xz() const {
        return span;
    }

    inline size_t size() {
        return span.size();
    }
    static ConvertedArrayXZ from_obj(const pybind11::handle &obj, const char *context_value_name);
    static ConvertedArrayXZ from_obj_or_int(
        const pybind11::handle &obj,
        size_t int_num_bits,
        const char *context_value_name,
        bool allow_twos_complement = false);
    static ConvertedArrayXZ from_obj_expecting_list(const pybind11::handle &obj, const char *context_value_name);

    ~ConvertedArrayXZ();
};

}  // namespace kickmix_py

#endif
