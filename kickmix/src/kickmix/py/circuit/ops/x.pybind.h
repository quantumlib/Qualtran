#ifndef KICKGEN_PYBIND_X_H
#define KICKGEN_PYBIND_X_H

#include <pybind11/pybind11.h>

#include "kickmix/py/circuit/circuit_builder.pybind.h"

namespace kickmix_py {

void broadcast_x_obj(PyCircuitBuilder &self, const pybind11::object &target_obj);

}  // namespace kickmix_py

#endif
