#ifndef KICKGEN_PYBIND_QCARRAY_H
#define KICKGEN_PYBIND_QCARRAY_H

#include <pybind11/pybind11.h>

#include "kickmix/id/array_xz.h"

namespace kickmix_py {

struct PyArrayXZ {
    kickmix::stride_span_xz view{};
    std::shared_ptr<kickmix::array_xz> owner{};
    bool operator==(const PyArrayXZ &other) const;
    static PyArrayXZ copy_of(const kickmix::stride_span_xz &other);
};

void register_qcarray_methods(pybind11::class_<PyArrayXZ> &c_qcarray);

}  // namespace kickmix_py

#endif
