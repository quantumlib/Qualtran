#ifndef KICKGEN_PYBIND_CCX_H
#define KICKGEN_PYBIND_CCX_H

#include <pybind11/pybind11.h>

#include "kickmix/py/circuit/circuit_builder.pybind.h"

namespace kickmix_py {

void broadcast_ccx_obj(
    PyCircuitBuilder &self,
    const pybind11::object &control2_obj,
    const pybind11::object &control1_obj,
    const pybind11::object &target_obj);
void broadcast_init_and_obj(
    PyCircuitBuilder &self,
    const pybind11::object &control1_obj,
    const pybind11::object &control2_obj,
    const pybind11::object &target_obj);
void broadcast_del_and_obj(
    PyCircuitBuilder &self,
    const pybind11::object &control1_obj,
    const pybind11::object &control2_obj,
    const pybind11::object &target_obj);

}  // namespace kickmix_py

#endif
