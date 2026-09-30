#include "cswap.pybind.h"

#include "kickmix/py/circuit/control_helper.pybind.h"
#include "kickmix/py/val/converted_array.pybind.h"

using namespace kickmix;
using namespace kickmix_py;

void kickmix_py::broadcast_cswap_obj(
    PyCircuitBuilder &self,
    const pybind11::object &control_obj,
    const pybind11::object &target1_obj,
    const pybind11::object &target2_obj) {
    auto target1_conv = ConvertedArrayXZ::from_obj(target1_obj, "target1");
    auto target2_conv = ConvertedArrayXZ::from_obj(target2_obj, "target2");
    auto target1 = target1_conv.span.checked_cast_to_qubit_ids("target1");
    auto target2 = target2_conv.span.checked_cast_to_qubit_ids("target2");
    if (target1.size() != target2.size()) {
        std::stringstream ss;
        ss << "cswap: len(target1) (" << target1.size() << ") != len(target2) (" << target2.size() << ")";
        throw std::invalid_argument(ss.str());
    }

    RaiiControlObjHelper control(self, control_obj, "control");
    if (control.skip) {
        return;
    }
    self.builder.broadcast_cswap(control, target1, target2);
}
