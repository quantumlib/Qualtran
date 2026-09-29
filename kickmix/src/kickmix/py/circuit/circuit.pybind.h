#ifndef KICKGEN_PYBIND_CIRCUIT_H
#define KICKGEN_PYBIND_CIRCUIT_H

#include <pybind11/pybind11.h>

#include "kickmix/circuit/circuit.h"

namespace kickmix {

void register_circuit_methods(pybind11::class_<kickmix::Circuit> &c_circuit);

}  // namespace kickmix

#endif
