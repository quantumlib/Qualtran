#ifndef KICKGEN_PYBIND_SWAP_H
#define KICKGEN_PYBIND_SWAP_H

#include "kickmix/py/circuit/circuit_builder.pybind.h"

namespace kickmix_py {

void broadcast_swap_obj(
    PyCircuitBuilder &self, const pybind11::object &target1_obj, const pybind11::object &target2_obj);

}  // namespace kickmix_py

#endif
