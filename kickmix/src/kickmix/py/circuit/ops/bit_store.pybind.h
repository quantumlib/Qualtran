#ifndef KICKGEN_PYBIND_BIT_STORE_H
#define KICKGEN_PYBIND_BIT_STORE_H

#include "kickmix/py/circuit/circuit_builder.pybind.h"

namespace kickmix_py {

void broadcast_bit_store_obj(
    PyCircuitBuilder &self,
    const pybind11::object &target_obj,
    const pybind11::object &value_obj,
    const pybind11::object &control_obj);

}  // namespace kickmix_py

#endif
