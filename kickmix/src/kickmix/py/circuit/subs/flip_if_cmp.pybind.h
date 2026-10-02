#ifndef KICKGEN_PYBIND_FLIP_IF_CMP_H
#define KICKGEN_PYBIND_FLIP_IF_CMP_H

#include "kickmix/py/circuit/circuit_builder.pybind.h"

namespace kickmix_py {

void append_flip_if_less_than_obj(
    PyCircuitBuilder &self,
    const pybind11::object &lhs_obj,
    const pybind11::object &rhs_obj,
    const pybind11::object &target_obj,
    const pybind11::object &or_equal_obj,
    const pybind11::object &control_obj,
    double btol);
void append_flip_if_equal_obj(
    PyCircuitBuilder &self,
    const pybind11::object &lhs_obj,
    const pybind11::object &rhs_obj,
    const pybind11::object &target_obj,
    const pybind11::object &control_obj);
void append_flip_if_greater_than_obj(
    PyCircuitBuilder &self,
    const pybind11::object &lhs_obj,
    const pybind11::object &rhs_obj,
    const pybind11::object &target_obj,
    const pybind11::object &or_equal_obj,
    const pybind11::object &control_obj,
    double btol);

}  // namespace kickmix_py

#endif
