#include "multi_control_x.pybind.h"

#include "kickmix/py/circuit/control_helper.pybind.h"
#include "kickmix/py/val/converted_array.pybind.h"

using namespace kickmix;
using namespace kickmix_py;

void multi_control_x(PyCircuitBuilder &self, const stride_span<const QubitId> &controls, QubitId target) {
    size_t n = controls.size();

    if (n == 0) {
        self.builder.x(target);
        return;
    }
    if (n == 1) {
        self.builder.cx(controls[0], target);
        return;
    }
    if (n == 2) {
        self.builder.ccx(controls[0], controls[1], target);
        return;
    }

    array_z anc_storage = array_z::alloc_noinit(n - 1);
    anc_storage[0] = controls[0];
    for (size_t k = 1; k < anc_storage.size(); k++) {
        anc_storage[k] = self.alloc_qubit();
    }
    stride_span<const QubitId> anc = anc_storage.operator stride_span_z().cast_data<QubitId>();
    self.builder.for_each(1, n - 1, [&](LoopBuilder &loop, iota k) {
        loop.reset(anc[k]);
        loop.ccx(controls[k], anc[k - 1], anc[k]);
    });
    self.builder.ccx(controls.back(), anc.back(), target);
    {
        auto b = self.builder.alloc_dirty_raii_bit();
        self.builder.for_each_reversed(1, n - 1, [&](LoopBuilder &loop, iota k) {
            loop.hmr(anc[k], b.bit);
            loop.cz_if(controls[k], anc[k - 1], b.bit);
        });
    }

    for (size_t k = 1; k < anc.size(); k++) {
        self.free_qubit(anc[k]);
    }
}

void kickmix_py::multi_control_x_obj(
    PyCircuitBuilder &self, const pybind11::object &controls_obj, const pybind11::object &target_obj) {
    auto controls_conv = ConvertedArrayXZ::from_obj_expecting_list(controls_obj, "controls");
    auto controls = controls_conv.span.checked_cast_to_qubit_or_bit_or_bool("controls");
    auto target_conv = ConvertedArrayXZ::from_obj(target_obj, "target");
    auto target = target_conv.span.checked_cast_to_qubit_ids("target");
    for (auto q : controls) {
        if (q.is_false()) {
            return;
        }
    }
    if (target.empty()) {
        return;
    }

    std::vector<QubitId> qubit_controls;
    qubit_controls.reserve(controls.size());
    size_t num_bit_controls = 0;
    for (auto e : controls) {
        if (e.is_qubit()) {
            qubit_controls.push_back((QubitId)e);
        } else if (e.is_bit()) {
            num_bit_controls += 1;
            self.builder.push_condition((BitId)e);
        }
    }

    if (qubit_controls.size() == 0) {
        self.builder.broadcast_x(target);
    } else if (qubit_controls.size() == 1) {
        self.builder.broadcast_cx(qubit_controls[0], target);
    } else {
        self.builder.broadcast_cx(target[0], target.skip(1));
        multi_control_x(self, qubit_controls, target[0]);
        self.builder.broadcast_cx(target[0], target.skip(1));
    }

    while (num_bit_controls--) {
        self.builder.pop_condition();
    }
}
