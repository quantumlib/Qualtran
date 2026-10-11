#ifndef KICKGEN_PYBIND_OPERATION_STATISTICS_H
#define KICKGEN_PYBIND_OPERATION_STATISTICS_H

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <pybind11/pybind11.h>
#include <string>
#include <string_view>

#include "kickmix/circuit/circuit.h"
#include "kickmix/util/fixed_precision_angle_128.h"

namespace kickmix_py {

struct PyInteractiveSimulator;

struct PyOperationStatistics {
    std::array<uint64_t, 256> raw_op_counts{};
    std::map<kickmix::FixedPrecisionAngle128, uint64_t> z_pow_angle_counts{};

    bool operator==(const PyOperationStatistics &other) const;
    bool operator!=(const PyOperationStatistics &other) const {
        return !(*this == other);
    }

    static PyOperationStatistics from_circuit(const kickmix::Circuit &circuit);
    static PyOperationStatistics from_simulator(const PyInteractiveSimulator &sim, std::optional<int64_t> shot_index);

    uint64_t ccp() const;
    uint64_t t() const;
    uint64_t rotations() const;
    uint64_t clifford_z_pow() const;
    uint64_t total() const;
    uint64_t get_item(std::string_view key) const;
    pybind11::dict z_pow_counts() const;
    pybind11::dict op_counts(const pybind11::object &z_pow_key) const;
    std::string repr() const;
};

pybind11::class_<PyOperationStatistics> register_operation_statistics_class(pybind11::module &m);

void register_operation_statistics_methods(pybind11::class_<PyOperationStatistics> &c_stats);

}  // namespace kickmix_py

#endif
