#ifndef KICKGEN_PYBIND_CZ_H
#define KICKGEN_PYBIND_CZ_H

#include "kickmix/py/circuit/circuit_builder.pybind.h"

namespace kickmix_py {

void broadcast_cz_obj(
    PyCircuitBuilder &self, const pybind11::object &control1_obj, const pybind11::object &control2_obj);

}  // namespace kickmix_py

#endif
