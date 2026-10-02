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
    bool ignore_debug_prints = false;
    bool count_operations = false;
    std::vector<kickmix::Sim<kickmix::b256, false>> simulators;
    std::vector<kickmix::Sim<kickmix::b256, true>> counting_simulators;
    std::vector<uint8_t> byte_buf;

    template <typename F>
    decltype(auto) with_sims(F &&fn) {
        return count_operations ? fn(counting_simulators) : fn(simulators);
    }
    template <typename F>
    decltype(auto) with_sims(F &&fn) const {
        return count_operations ? fn(counting_simulators) : fn(simulators);
    }

    void ensure_byte_buf_can_store_bits(size_t num_bits);
    void clear_op_counts();
    void set_ignore_debug_prints(bool value);
    PyInteractiveSimulator(size_t batch_size, bool ignore_debug_prints = false, bool count_operations = false);
    void ensure_big_enough_state_for(size_t num_qubits, size_t num_bits);
    void ensure_big_enough_state_for(kickmix::QubitOrBit val);
};

pybind11::class_<PyInteractiveSimulator> register_interactive_simulator_class(pybind11::module &m);

void register_interactive_simulator_methods(pybind11::class_<PyInteractiveSimulator> &c_sim);

}  // namespace kickmix_py

#endif
