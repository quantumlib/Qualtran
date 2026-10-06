#ifndef KICKGEN_PYBIND_MULTI_CONTROL_X_H
#define KICKGEN_PYBIND_MULTI_CONTROL_X_H

#include "kickmix/py/circuit/circuit_builder.pybind.h"

namespace kickmix_py {

void multi_control_x_obj(
    PyCircuitBuilder &self, const pybind11::object &controls_obj, const pybind11::object &target_obj);

}  // namespace kickmix_py

#endif
