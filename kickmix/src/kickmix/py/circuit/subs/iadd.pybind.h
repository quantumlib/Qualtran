#ifndef KICKGEN_PYBIND_IADD_H
#define KICKGEN_PYBIND_IADD_H

#include <string_view>

#include "kickmix/py/circuit/circuit_builder.pybind.h"

namespace kickmix_py {

void append_iadd_obj(
    PyCircuitBuilder &self,
    const pybind11::object &offset,
    const pybind11::object &target,
    const pybind11::object &control,
    double btol);

void append_isub_obj(
    PyCircuitBuilder &self,
    const pybind11::object &offset,
    const pybind11::object &target,
    const pybind11::object &control,
    double btol);

}  // namespace kickmix_py

#endif
