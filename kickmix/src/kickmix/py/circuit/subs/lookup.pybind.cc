#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>

#include "kickmix/gen/lookup/gen_lookup.h"
#include "kickmix/id/qubit_or_true.h"
#include "kickmix/py/circuit/circuit.pybind.h"
#include "kickmix/py/circuit/circuit_builder.pybind.h"
#include "kickmix/py/val/converted_array.pybind.h"
#include "kickmix/py/val/id.pybind.h"

using namespace kickmix;
using namespace kickmix_py;

void kickmix_py::append_init_lookup(
    PyCircuitBuilder &self,
    const pybind11::object &table,
    const pybind11::object &address_obj,
    const pybind11::object &target_obj) {
    ConvertedArrayXZ address = ConvertedArrayXZ::from_obj(address_obj, "address");
    ConvertedArrayXZ target = ConvertedArrayXZ::from_obj(target_obj, "target");
    if (address.size() > 32) {
        throw std::invalid_argument("len(address) > 32");
    }

    array_z clean = array_z::alloc_noinit(address.size(), QXZTypeTag8::QUBIT_ID);
    for (size_t k = 0; k < address.size(); k++) {
        clean[k] = self.alloc_qubit();
    }
    std::span<const QubitId> clean_span{(QubitId *)clean.items, address.size()};
    array_z table_bits = read_python_int_list_into_table_data(table, target.size());
    if (table_bits.size() != target.size() << address.size()) {
        throw std::invalid_argument("table contents didn't match the expected size of the lookup.");
    }
    auto target_qubits = target.span.checked_cast_to_qubit_ids("target");

    self.builder.broadcast_reset(target_qubits);
    gen_lookup(
        self.builder,
        CircuitGenCtx{clean_span},
        table_bits,
        address.span.checked_cast_to_qubit_ids("address"),
        target_qubits,
        'X',
        true);

    for (size_t k = address.size(); k--;) {
        self.free_qubit((QubitId)clean[k]);
    }
}

void kickmix_py::append_del_lookup(
    PyCircuitBuilder &self,
    const pybind11::object &table,
    const pybind11::object &address_obj,
    const pybind11::object &target_obj) {
    ConvertedArrayXZ address = ConvertedArrayXZ::from_obj(address_obj, "address");
    ConvertedArrayXZ target = ConvertedArrayXZ::from_obj(target_obj, "target");
    if (address.size() > 32) {
        throw std::invalid_argument("len(address) > 32");
    }

    array_z clean = array_z::alloc_noinit(address.size(), QXZTypeTag8::QUBIT_ID);
    for (size_t k = 0; k < address.size(); k++) {
        clean[k] = self.alloc_qubit();
    }
    std::span<const QubitId> clean_span{(QubitId *)clean.items, address.size()};
    array_z table_bits = read_python_int_list_into_table_data(table, target.size());
    if (table_bits.size() != target.size() << address.size()) {
        throw std::invalid_argument("table contents didn't match the expected size of the lookup.");
    }

    gen_unlookup(
        self.builder,
        CircuitGenCtx{clean_span},
        table_bits,
        address.span.checked_cast_to_qubit_ids("address"),
        target.span.checked_cast_to_qubit_ids("target"));

    for (size_t k = address.size(); k--;) {
        self.free_qubit((QubitId)clean[k]);
    }
}
