#ifndef KICKGEN_PYBIND_BIT_ROTATE_H
#define KICKGEN_PYBIND_BIT_ROTATE_H

#include "kickmix/py/circuit/circuit_builder.pybind.h"

namespace kickmix_py {

void append_left_rotate_obj(PyCircuitBuilder &self, const pybind11::object &target, const pybind11::object &control);
void append_right_rotate_obj(PyCircuitBuilder &self, const pybind11::object &target, const pybind11::object &control);

}  // namespace kickmix_py

#endif
