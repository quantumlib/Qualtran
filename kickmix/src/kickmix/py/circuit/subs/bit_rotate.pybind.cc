#include "bit_rotate.pybind.h"

#include "kickmix/py/circuit/control_helper.pybind.h"
#include "kickmix/py/val/converted_array.pybind.h"

using namespace kickmix;
using namespace kickmix_py;

static void controlled_reverse(CircuitBuilder &builder, stride_span<const QubitId> target, QubitOrTrue control = true) {
    size_t n = target.size();
    size_t h = n >> 1;
    for (size_t k = 0; k < h; k++) {
        builder.cswap(control, target[k], target[n - k - 1]);
    }
}

static void controlled_shifted_reverse(
    CircuitBuilder &builder, stride_span<const QubitId> target, int64_t offset, QubitOrTrue control = true) {
    if (target.empty()) {
        return;
    }
    size_t n = target.size();
    offset %= (int64_t)n;
    offset += n;
    offset %= (int64_t)n;
    controlled_reverse(builder, target.keep((size_t)offset), control);
    controlled_reverse(builder, target.skip((size_t)offset), control);
}

struct RaiiStoreXor {
    CircuitBuilder *builder;
    QubitOrBitOrBool v1;
    QubitOrBitOrBool v2;
    QubitOrBitOrBool result;
    QubitOrBitOrBool apply() {
        if (v1.is_qubit()) {
            if (v2.is_qubit()) {
                builder->cx((QubitId)v1, (QubitId)v2);
                return v2;
            } else if (v2.is_bit()) {
                builder->cx((BitId)v2, (QubitId)v1);
                return v1;
            } else {
                builder->cx((bool)v2, (QubitId)v1);
                return v1;
            }
        } else if (v1.is_bit()) {
            if (v2.is_qubit()) {
                builder->cx((BitId)v1, (QubitId)v2);
                return v2;
            } else if (v2.is_bit()) {
                builder->cx((BitId)v1, (BitId)v2);
                return v2;
            } else {
                builder->cx((bool)v2, (BitId)v1);
                return v1;
            }
        } else {
            if (v2.is_qubit()) {
                builder->cx((bool)v1, (QubitId)v2);
                return v2;
            } else if (v2.is_bit()) {
                builder->cx((bool)v1, (BitId)v2);
                return v2;
            } else {
                return (bool)v1 != (bool)v2;
            }
        }
    }
    RaiiStoreXor(CircuitBuilder &builder, QubitOrBitOrBool v1, QubitOrBitOrBool v2)
        : builder(&builder), v1(v1), v2(v2) {
        result = apply();
    }
    ~RaiiStoreXor() {
        apply();
    }
};

template <typename TCallback>
static void if_and_do(PyCircuitBuilder &self, QubitOrBitOrBool v1, QubitOrBitOrBool v2, TCallback callback) {
    if (v2 == false || v1 == false) {
        return;
    }
    if (v1.is_bit()) {
        self.builder.push_condition((BitId)v1);
    }
    if (v2.is_bit()) {
        self.builder.push_condition((BitId)v2);
    }
    QubitOrTrue control = true;
    if (v1.is_qubit() && v2.is_qubit()) {
        control = self.alloc_qubit();
        self.builder.reset((QubitId)control);
        self.builder.ccx((QubitId)v1, (QubitId)v2, (QubitId)control);
    } else if (v1.is_qubit()) {
        control = (QubitId)v1;
    } else if (v2.is_qubit()) {
        control = (QubitId)v2;
    }
    callback(control);
    if (v1.is_qubit() && v2.is_qubit()) {
        self.builder.del_and((QubitId)v1, (QubitId)v2, (QubitId)control);
        self.free_qubit((QubitId)control);
    }
    if (v1.is_bit()) {
        self.builder.pop_condition();
    }
    if (v2.is_bit()) {
        self.builder.pop_condition();
    }
}

static void controlled_fixed_left_rotate(
    CircuitBuilder &builder, stride_span<const QubitId> target, int64_t shift, QubitOrTrue control) {
    if (target.empty()) {
        return;
    }
    shift %= (int64_t)target.size();
    shift += target.size();
    shift %= target.size();
    if (shift == 0) {
        return;
    }
    controlled_shifted_reverse(builder, target, 0, control);
    controlled_shifted_reverse(builder, target, shift, control);
}

static void decomposed_left_rotate(
    PyCircuitBuilder &self,
    stride_span<const QubitId> target,
    stride_span_z shift,
    QubitOrTrue control,
    bool inverted) {
    if (shift.size() > 60) {
        throw std::invalid_argument("len(shift) > 60");
    }
    for (size_t k = 0; k <= shift.size(); k++) {
        QubitOrBitOrBool c0 = k > 0 ? shift[k - 1] : false;
        QubitOrBitOrBool c1 = k < shift.size() ? shift[k] : false;
        RaiiStoreXor parity(self.builder, c0, c1);
        if_and_do(self, parity.result, control, [&](QubitOrTrue merged_control) {
            controlled_shifted_reverse(self.builder, target, int64_t{inverted ? -1 : +1} << k, merged_control);
        });
    }
}

void kickmix_py::append_left_rotate_obj(
    PyCircuitBuilder &self,
    const pybind11::object &target_obj,
    const pybind11::object &shift_obj,
    const pybind11::object &control_obj) {
    auto target_conv = ConvertedArrayXZ::from_obj_expecting_list(target_obj, "target");
    auto target = target_conv.span.checked_cast_to_qubit_ids("target");
    RaiiControlObjHelper control(self, control_obj, "control");
    if (control.skip) {
        return;
    }

    auto mark = self.builder.raii_mark_block_entry("left_rotate");
    if (pybind11::isinstance<pybind11::int_>(shift_obj)) {
        int64_t shift = pybind11::cast<int64_t>(shift_obj);
        controlled_fixed_left_rotate(self.builder, target, shift, control);
    } else {
        auto shift_conv = ConvertedArrayXZ::from_obj_expecting_list(shift_obj, "shift");
        auto shift = shift_conv.span.checked_cast_to_qubit_or_bit_or_bool("shift");
        decomposed_left_rotate(self, target, shift, control, false);
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
    if (control.skip) {
        return;
    }

    auto mark = self.builder.raii_mark_block_entry("right_rotate");
    if (pybind11::isinstance<pybind11::int_>(shift_obj)) {
        int64_t shift = pybind11::cast<int64_t>(shift_obj);
        controlled_fixed_left_rotate(self.builder, target, -shift, control);
    } else {
        auto shift_conv = ConvertedArrayXZ::from_obj_expecting_list(shift_obj, "shift");
        auto shift = shift_conv.span.checked_cast_to_qubit_or_bit_or_bool("shift");
        decomposed_left_rotate(self, target, shift, control, true);
    }
}
