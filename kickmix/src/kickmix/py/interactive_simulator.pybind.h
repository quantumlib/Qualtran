#ifndef KICKGEN_PYBIND_INTERACTIVE_SIMULATOR_H
#define KICKGEN_PYBIND_INTERACTIVE_SIMULATOR_H

#include <pybind11/pybind11.h>

#include "kickmix/sim/sim.h"
#include "kickmix/simd/simd.h"

namespace kickmix_py {

constexpr size_t PY_SIM_WORD_BITS = 256;

struct PyInteractiveSimulator {
    std::vector<kickmix::RegisterData> register_data;
    size_t batch_size;
    std::vector<kickmix::Sim<kickmix::b256, false>> simulators;
    std::vector<uint8_t> byte_buf;

    void ensure_byte_buf_can_store_bits(size_t num_bits);
    PyInteractiveSimulator(size_t batch_size);
    void ensure_big_enough_state_for(size_t num_qubits, size_t num_bits);
    void ensure_big_enough_state_for(kickmix::QubitOrBit val);
};

pybind11::class_<PyInteractiveSimulator> register_interactive_simulator_class(pybind11::module &m);

void register_interactive_simulator_methods(pybind11::class_<PyInteractiveSimulator> &c_sim);

}  // namespace kickmix_py

#endif
