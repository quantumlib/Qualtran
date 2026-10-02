#ifndef KICKGEN_PYBIND_CIRCUIT_BUILDER_H
#define KICKGEN_PYBIND_CIRCUIT_BUILDER_H

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kickmix/build/circuit_builder.h"
#include "kickmix/py/val/array.pybind.h"
#include "kickmix/py/val/id.pybind.h"

namespace kickmix_py {

struct PyCircuitBuilder {
    kickmix::CircuitBuilder builder;
    std::vector<kickmix::QubitId> free_qids;
    std::vector<kickmix::BitId> free_bids;
    std::vector<bool> qubit_allocations;
    std::vector<bool> bit_allocations;
    size_t max_qubits = SIZE_MAX;
    size_t allocated_qubits = 0;
    size_t allocated_bits = 0;
    std::vector<std::vector<uint32_t>> scope_qubit_allocations;
    std::vector<std::vector<uint32_t>> scope_bit_allocations;

    void start_auto_free_scope() {
        scope_qubit_allocations.push_back({});
        scope_bit_allocations.push_back({});
    }
    void end_auto_free_scope() {
        if (!scope_qubit_allocations.empty()) {
            while (!scope_qubit_allocations.back().empty()) {
                uint32_t id = scope_qubit_allocations.back().back();
                scope_qubit_allocations.back().pop_back();
                if (id < qubit_allocations.size() && qubit_allocations[id]) {
                    free_qids.push_back(kickmix::QubitId{id});
                    qubit_allocations[id] = false;
                }
                allocated_qubits -= 1;
            }
            scope_qubit_allocations.pop_back();
        }
        if (!scope_bit_allocations.empty()) {
            while (!scope_bit_allocations.back().empty()) {
                uint32_t id = scope_bit_allocations.back().back();
                scope_bit_allocations.back().pop_back();
                if (id < bit_allocations.size() && bit_allocations[id]) {
                    free_qids.push_back(kickmix::QubitId{id});
                    bit_allocations[id] = false;
                }
                allocated_bits -= 1;
            }
            scope_bit_allocations.pop_back();
        }
    }

    kickmix::QubitId alloc_qubit() {
        kickmix::QubitId result;
        if (free_qids.empty()) {
            result = kickmix::QubitId(builder.next_qubit_id++);
        } else {
            result = free_qids.back();
            free_qids.pop_back();
        }
        while (qubit_allocations.size() <= result.untagged_id()) {
            qubit_allocations.push_back(false);
        }
        qubit_allocations[result.untagged_id()] = true;
        if (!scope_qubit_allocations.empty()) {
            scope_qubit_allocations.back().push_back(result.untagged_id());
        }
        allocated_qubits += 1;
        return result;
    }

    kickmix::BitId alloc_bit() {
        kickmix::BitId result;
        if (free_bids.empty()) {
            result = kickmix::BitId(builder.next_bit_id++);
        } else {
            result = free_bids.back();
            free_bids.pop_back();
        }
        while (bit_allocations.size() <= result.untagged_id()) {
            bit_allocations.push_back(false);
        }
        bit_allocations[result.untagged_id()] = true;
        if (!scope_bit_allocations.empty()) {
            scope_bit_allocations.back().push_back(result.untagged_id());
        }
        allocated_bits += 1;
        return result;
    }

    void free_qubit(kickmix::QubitId id) {
        if (id.untagged_id() >= qubit_allocations.size() || !qubit_allocations[id.untagged_id()]) {
            std::stringstream ss;
            ss << "Freed a qubit that wasn't allocated (or was already freed): ";
            ss << "q(" << id << ")";
            throw std::invalid_argument(ss.str());
        }
        qubit_allocations[id.untagged_id()] = false;
        free_qids.push_back(kickmix::QubitId{id});
        allocated_qubits -= 1;
    }

    void free_bit(kickmix::BitId id) {
        if (id.untagged_id() >= bit_allocations.size() || !bit_allocations[id.untagged_id()]) {
            std::stringstream ss;
            ss << "Freed a bit that wasn't allocated (or was already freed): ";
            ss << "b(" << id.untagged_id() << ")";
            throw std::invalid_argument(ss.str());
        }
        bit_allocations[id.untagged_id()] = false;
        free_bids.push_back(id);
        allocated_bits -= 1;
    }

    size_t num_free_qubits() const {
        return max_qubits - allocated_qubits;
    }
};

void register_circuit_builder_methods(pybind11::class_<PyCircuitBuilder> &c_circuit_builder);

kickmix::array_z read_python_int_list_into_table_data(const pybind11::object &table, size_t word_length);

void append_init_lookup(
    PyCircuitBuilder &self,
    const pybind11::object &table,
    const pybind11::object &address_obj,
    const pybind11::object &target_obj);
void append_del_lookup(
    PyCircuitBuilder &self,
    const pybind11::object &table,
    const pybind11::object &address_obj,
    const pybind11::object &target_obj);

}  // namespace kickmix_py

#endif
