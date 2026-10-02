#ifndef KICKGEN_qc_stride_span_x_H
#define KICKGEN_qc_stride_span_x_H

#include "kickmix/mem/stride_span.h"
#include "qubit_or_bit_or_bool.h"
#include "qubit_or_xbit_or_xbool.h"
#include "qubit_or_xzbit_or_xzbool.h"

namespace kickmix {

struct stride_span_x {
    const QubitOrXBitOrXBool *ptr;
    int64_t stride;
    size_t count;
    QXZTypeTag8 common_type;

    inline stride_span_x() : ptr(nullptr), stride(0), count(0), common_type(QXZTypeTag8::QUBIT_ID) {
    }
    inline stride_span_x(const QubitOrXBitOrXBool *ptr, int64_t stride, size_t count, QXZTypeTag8 common_type)
        : ptr(ptr), stride(stride), count(count), common_type(common_type) {
    }
    inline stride_span_x(stride_span<const QubitId> v)
        : ptr((const QubitOrXBitOrXBool *)v.data()),
          stride(v.stride),
          count(v.size()),
          common_type(QXZTypeTag8::QUBIT_ID) {
    }
    inline stride_span_x(stride_span<const XBitId> v)
        : ptr((const QubitOrXBitOrXBool *)v.data()),
          stride(v.stride),
          count(v.size()),
          common_type(QXZTypeTag8::XBIT_ID) {
    }
    inline stride_span_x(stride_span<const XBool> v)
        : ptr((const QubitOrXBitOrXBool *)v.data()),
          stride(v.stride),
          count(v.size()),
          common_type(QXZTypeTag8::XBOOL_VAL) {
    }
    inline stride_span_x(stride_span<const QubitOrXBitOrXBool> v)
        : ptr(v.data()), stride(v.stride), count(v.size()), common_type(QXZTypeTag8::MAY_BE_MIXED_BUT_IS_ALL_XLIKE) {
    }
    template <typename T>
    stride_span<const T> cast_data() const {
        return {(const T *)ptr, stride, count};
    }
};

}  // namespace kickmix

#endif
