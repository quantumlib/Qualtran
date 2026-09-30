#include "bit_rotate.pybind.h"

#include "kickmix/py/circuit/control_helper.pybind.h"
#include "kickmix/py/val/converted_array.pybind.h"

using namespace kickmix;
using namespace kickmix_py;

void kickmix_py::append_left_rotate_obj(
    PyCircuitBuilder &self, const pybind11::object &target_obj, const pybind11::object &control_obj) {
    auto target_conv = ConvertedArrayXZ::from_obj_expecting_list(target_obj, "target");
    auto target = target_conv.span.checked_cast_to_qubit_ids("target");
    RaiiControlObjHelper control(self, control_obj, "control");
    if (control.skip) {
        return;
    }
    self.builder.cleft_rotate(control, target);
}

void kickmix_py::append_right_rotate_obj(
    PyCircuitBuilder &self, const pybind11::object &target_obj, const pybind11::object &control_obj) {
    auto target_conv = ConvertedArrayXZ::from_obj_expecting_list(target_obj, "target");
    auto target = target_conv.span.checked_cast_to_qubit_ids("target");
    RaiiControlObjHelper control(self, control_obj, "control");
    if (control.skip) {
        return;
    }
    self.builder.cright_rotate(control, target);
}
