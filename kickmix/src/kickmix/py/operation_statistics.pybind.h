#ifndef KICKGEN_PYBIND_OPERATION_STATISTICS_H
#define KICKGEN_PYBIND_OPERATION_STATISTICS_H

#include <array>
#include <cstdint>
#include <optional>
#include <pybind11/pybind11.h>
#include <string>
#include <string_view>

#include "kickmix/circuit/circuit.h"

namespace kickmix_py {

struct PyInteractiveSimulator;

struct PyOperationStatistics {
    std::array<uint64_t, 256> raw_op_counts{};
    std::array<uint64_t, 129> z_pow_level_counts{};

    bool operator==(const PyOperationStatistics &other) const = default;
    bool operator!=(const PyOperationStatistics &other) const = default;

    static PyOperationStatistics from_circuit(const kickmix::Circuit &circuit);
    static PyOperationStatistics from_simulator(const PyInteractiveSimulator &sim, std::optional<int64_t> shot_index);

    uint64_t num_toffoli_like() const;
    uint64_t num_t_like() const;
    uint64_t num_instructions() const;
    uint64_t num_z_pow(int64_t min_level, std::optional<int64_t> max_level) const;
    pybind11::dict z_pow_clifford_hierarchy_counts() const;
    uint64_t get_item(std::string_view key) const;
    pybind11::object get_with_default(std::string_view key, const pybind11::object &default_val) const;
    bool contains(const pybind11::object &key) const;
    size_t len() const;
    pybind11::list keys() const;
    pybind11::list values() const;
    pybind11::list items() const;
    std::string repr() const;
};

pybind11::class_<PyOperationStatistics> register_operation_statistics_class(pybind11::module &m);

void register_operation_statistics_methods(pybind11::class_<PyOperationStatistics> &c_stats);

}  // namespace kickmix_py

#endif
