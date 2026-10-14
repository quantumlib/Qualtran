#include "bit_rotate.pybind.h"

#include "kickmix/gen/bits/gen_bit_rotate.h"
#include "kickmix/py/circuit/control_helper.pybind.h"
#include "kickmix/py/val/converted_array.pybind.h"

using namespace kickmix;
using namespace kickmix_py;

struct RaiiCtxHelper {
    PyCircuitBuilder *builder;
    QubitOrFalse anc;
    bool failed;
    CircuitGenCtx ctx;
    RaiiCtxHelper(PyCircuitBuilder *builder, bool need_qubit) : builder(builder), anc(), failed(), ctx() {
        if (need_qubit) {
            failed = builder->num_free_qubits() == 0;
            if (!failed) {
                anc = builder->alloc_qubit();
                ctx.clean_workspace = {(QubitId *)&anc, 1};
            }
        }
    }
    ~RaiiCtxHelper() {
        if (builder != nullptr) {
            if (anc.is_qubit()) {
                builder->free_qubit(anc.qubit());
            }
            builder = nullptr;
        }
    }
};

void kickmix_py::append_left_rotate_obj(
    PyCircuitBuilder &self,
    const pybind11::object &target_obj,
    const pybind11::object &shift_obj,
    const pybind11::object &control_obj) {
    auto target_conv = ConvertedArrayXZ::from_obj_expecting_list(target_obj, "target");
    auto target = target_conv.span.checked_cast_to_qubit_ids("target");
    RaiiControlObjHelper control(self, control_obj, "control");

    auto mark = self.builder.raii_mark_block_entry("left_rotate");
    if (control.skip) {
        return;
    }

    if (pybind11::isinstance<pybind11::int_>(shift_obj)) {
        int64_t shift = pybind11::cast<int64_t>(shift_obj);
        gen_left_rotate(self.builder, CircuitGenCtx{}, target, shift, control.value, "");
    } else {
        auto shift_conv = ConvertedArrayXZ::from_obj_expecting_list(shift_obj, "shift");
        auto shift = shift_conv.span.checked_cast_to_qubit_or_bit_or_bool("shift");
        RaiiCtxHelper helper(&self, shift.has_any_qubits() && control.value.is_qubit());
        if (helper.failed) {
            std::stringstream ss;
            ss << "Need 1 clean qubit for left_rotate because control=";
            ss << pybind11::repr(control_obj);
            ss << " is a qubit and shift=";
            ss << pybind11::repr(shift_obj) << " has qubits (but no clean qubits are available).";
            throw std::invalid_argument(ss.str());
        }
        gen_left_rotate(self.builder, helper.ctx, target, shift, control.value, false, "");
    }
}

void kickmix_py::append_right_rotate_obj(
    PyCircuitBuilder &self,
    const pybind11::object &target_obj,
    const pybind11::object &shift_obj,
    const pybind11::object &control_obj) {
    auto target_conv = ConvertedArrayXZ::from_obj_expecting_list(target_obj, "target");
    auto target = target_conv.span.checked_cast_to_qubit_ids("target");
    RaiiControlObjHelper control(self, control_obj, "control");

    auto mark = self.builder.raii_mark_block_entry("right_rotate");
    if (control.skip) {
        return;
    }

    if (pybind11::isinstance<pybind11::int_>(shift_obj)) {
        int64_t shift = pybind11::cast<int64_t>(shift_obj);
        gen_right_rotate(self.builder, CircuitGenCtx{}, target, shift, control.value, "");
    } else {
        auto shift_conv = ConvertedArrayXZ::from_obj_expecting_list(shift_obj, "shift");
        auto shift = shift_conv.span.checked_cast_to_qubit_or_bit_or_bool("shift");
        RaiiCtxHelper helper(&self, shift.has_any_qubits() && control.value.is_qubit());
        if (helper.failed) {
            std::stringstream ss;
            ss << "Need 1 clean qubit for right_rotate because control=";
            ss << pybind11::repr(control_obj);
            ss << " is a qubit and shift=";
            ss << pybind11::repr(shift_obj) << " has qubits (but no clean qubits are available).";
            throw std::invalid_argument(ss.str());
        }
        gen_right_rotate(self.builder, helper.ctx, target, shift, control.value, false, "");
    }
}
