#ifndef KICKGEN_PYBIND_CCZ_H
#define KICKGEN_PYBIND_CCZ_H

#include <pybind11/pybind11.h>

#include "kickmix/py/circuit/circuit_builder.pybind.h"

namespace kickmix_py {

void broadcast_ccz_obj(
    PyCircuitBuilder &self,
    const pybind11::object &control1_obj,
    const pybind11::object &control2_obj,
    const pybind11::object &control3_obj);

}  // namespace kickmix_py

#endif
