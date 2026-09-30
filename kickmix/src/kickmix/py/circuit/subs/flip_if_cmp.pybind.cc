#include "flip_if_cmp.pybind.h"

#include "kickmix/gen/comparators/gen_cmp.h"
#include "kickmix/py/circuit/control_helper.pybind.h"
#include "kickmix/py/val/converted_array.pybind.h"

using namespace kickmix;
using namespace kickmix_py;

void kickmix_py::append_flip_if_less_than_obj(
    PyCircuitBuilder &self,
    const pybind11::object &lhs_obj,
    const pybind11::object &rhs_obj,
    const pybind11::object &target_obj,
    const pybind11::object &or_equal_obj,
    const pybind11::object &control_obj,
    double btol) {
    auto lhs_conv = ConvertedArrayXZ::from_obj(lhs_obj, "lhs");
    auto lhs = lhs_conv.span.checked_cast_to_qubit_ids("lhs");
    auto rhs_conv = ConvertedArrayXZ::from_obj_or_int(rhs_obj, lhs.size(), "rhs");
    auto rhs = rhs_conv.span.checked_cast_to_qubit_or_bit_or_bool("rhs");
    QubitOrBitOrBool or_equal = obj_to_qubit_or_bit_or_bool(or_equal_obj, "or_equal");
    RaiiXControlObjHelper target(self, target_obj, "target");
    RaiiControlObjHelper control(self, control_obj, "control");
    if (control.skip || target.skip) {
        return;
    }

    array_z clean = array_z::alloc_noinit(std::min(self.num_free_qubits(), lhs.size()), QXZTypeTag8::QUBIT_ID);
    for (size_t k = 0; k < clean.size(); k++) {
        clean[k] = self.alloc_qubit();
    }
    std::span<const QubitId> clean_span{(QubitId *)clean.items, clean.size()};
    gen_flip_if_lt(self.builder, CircuitGenCtx{clean_span}, lhs, rhs, target, or_equal, control, btol);
    for (size_t k = clean.size(); k--;) {
        self.free_qubit((QubitId)clean[k]);
    }
}

void kickmix_py::append_flip_if_equal_obj(
    PyCircuitBuilder &self,
    const pybind11::object &lhs_obj,
    const pybind11::object &rhs_obj,
    const pybind11::object &target_obj,
    const pybind11::object &control_obj) {
    auto lhs_conv = ConvertedArrayXZ::from_obj(lhs_obj, "lhs");
    auto lhs = lhs_conv.span.checked_cast_to_qubit_ids("lhs");
    auto rhs_conv = ConvertedArrayXZ::from_obj_or_int(rhs_obj, lhs.size(), "rhs");
    auto rhs = rhs_conv.span.checked_cast_to_qubit_or_bit_or_bool("rhs");
    RaiiXControlObjHelper target(self, target_obj, "target");
    RaiiControlObjHelper control(self, control_obj, "control");
    if (control.skip || target.skip) {
        return;
    }

    array_z clean = array_z::alloc_noinit(std::min(self.num_free_qubits(), lhs.size()), QXZTypeTag8::QUBIT_ID);
    for (size_t k = 0; k < clean.size(); k++) {
        clean[k] = self.alloc_qubit();
    }
    std::span<const QubitId> clean_span{(QubitId *)clean.items, clean.size()};
    gen_flip_if_eq(self.builder, CircuitGenCtx{clean_span}, lhs, rhs, target, control);
    for (size_t k = clean.size(); k--;) {
        self.free_qubit((QubitId)clean[k]);
    }
}

void kickmix_py::append_flip_if_greater_than_obj(
    PyCircuitBuilder &self,
    const pybind11::object &lhs_obj,
    const pybind11::object &rhs_obj,
    const pybind11::object &target_obj,
    const pybind11::object &or_equal_obj,
    const pybind11::object &control_obj,
    double btol) {
    auto lhs_conv = ConvertedArrayXZ::from_obj(lhs_obj, "lhs");
    auto lhs = lhs_conv.span.checked_cast_to_qubit_ids("lhs");
    auto rhs_conv = ConvertedArrayXZ::from_obj_or_int(rhs_obj, lhs.size(), "rhs");
    auto rhs = rhs_conv.span.checked_cast_to_qubit_or_bit_or_bool("rhs");
    QubitOrBitOrBool or_equal = obj_to_qubit_or_bit_or_bool(or_equal_obj, "or_equal");
    RaiiXControlObjHelper target(self, target_obj, "target");
    RaiiControlObjHelper control(self, control_obj, "control");
    if (control.skip || target.skip) {
        return;
    }

    array_z clean = array_z::alloc_noinit(std::min(self.num_free_qubits(), lhs.size()), QXZTypeTag8::QUBIT_ID);
    for (size_t k = 0; k < clean.size(); k++) {
        clean[k] = self.alloc_qubit();
    }
    std::span<const QubitId> clean_span{(QubitId *)clean.items, clean.size()};
    gen_flip_if_gt(self.builder, CircuitGenCtx{clean_span}, lhs, rhs, target, or_equal, control, btol);
    for (size_t k = clean.size(); k--;) {
        self.free_qubit((QubitId)clean[k]);
    }
}
