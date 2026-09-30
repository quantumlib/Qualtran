#include "ccx.pybind.h"

#include <kickmix/py/val/broadcast_resolver.pybind.h>

using namespace kickmix;
using namespace kickmix_py;

static inline void broadcast_ccx_case(
    PyCircuitBuilder &self,
    stride_ptr<const bool> c1,
    stride_ptr<const bool> c2,
    stride_ptr<const QubitId> t,
    size_t n) {
    QubitId *t_out = (QubitId *)self.builder.mut.q0.grab_writeable(n);
    QubitId *t_out_start = t_out;
    for (size_t k = 0; k < n; k++) {
        *t_out = *t++;
        bool b1 = *c1++;
        bool b2 = *c2++;
        bool keep = b1 && b2;
        t_out += keep;
    }
    self.builder.mut.q0.rewind_tail((uint32_t *)t_out);
    self.builder.mut.op_types.push_back_repeat(OpType::X, t_out - t_out_start);
}

static inline void broadcast_ccx_case(
    PyCircuitBuilder &self,
    stride_ptr<const bool> c1,
    stride_ptr<const BitId> c2,
    stride_ptr<const QubitId> t,
    size_t n) {
    QubitId *t_out = (QubitId *)self.builder.mut.q0.grab_writeable(n);
    BitId *cond_out = (BitId *)self.builder.mut.bc.grab_writeable(n);
    QubitId *t_out_start = t_out;
    for (size_t k = 0; k < n; k++) {
        *t_out = *t++;
        *cond_out = *c2++;
        bool keep = *c1++ != 0;
        t_out += keep;
        cond_out += keep;
    }
    self.builder.mut.q0.rewind_tail((uint32_t *)t_out);
    self.builder.mut.bc.rewind_tail((uint32_t *)cond_out);
    self.builder.mut.op_types.push_back_repeat(OpType::X_IF, t_out - t_out_start);
}

static inline void broadcast_ccx_case(
    PyCircuitBuilder &self,
    stride_ptr<const bool> c1,
    stride_ptr<const QubitId> c2,
    stride_ptr<const QubitId> t,
    size_t n) {
    QubitId *c_out = (QubitId *)self.builder.mut.qq0.grab_writeable(n);
    QubitId *t_out = (QubitId *)self.builder.mut.qq1.grab_writeable(n);
    QubitId *t_out_start = t_out;
    for (size_t k = 0; k < n; k++) {
        *c_out = *c2++;
        *t_out = *t++;
        bool keep = *c1++ != 0;
        c_out += keep;
        t_out += keep;
    }
    self.builder.mut.qq0.rewind_tail((uint32_t *)c_out);
    self.builder.mut.qq1.rewind_tail((uint32_t *)t_out);
    self.builder.mut.op_types.push_back_repeat(OpType::CX, t_out - t_out_start);
}

static inline void broadcast_ccx_case(
    PyCircuitBuilder &self,
    stride_ptr<const BitId> c1,
    stride_ptr<const BitId> c2,
    stride_ptr<const QubitId> t,
    size_t n) {
    self.builder.pattern.pattern_push_condition(c2);
    self.builder.pattern.pattern_x_if(t, c1);
    self.builder.pattern.pattern_pop_condition();
    self.builder.pattern.dump_into(self.builder.mut, 0, n, false);
}

static inline void broadcast_ccx_case(
    PyCircuitBuilder &self,
    stride_ptr<const BitId> c1,
    stride_ptr<const QubitId> c2,
    stride_ptr<const QubitId> t,
    size_t n) {
    self.builder.pattern.pattern_cx_if(c2, t, c1);
    self.builder.pattern.dump_into(self.builder.mut, 0, n, false);
}

static inline void broadcast_ccx_case(
    PyCircuitBuilder &self,
    stride_ptr<const QubitId> c2,
    stride_ptr<const QubitId> c1,
    stride_ptr<const QubitId> t,
    size_t n) {
    self.builder.mut.qqq0.push_back_many({(const uint32_t *)c2.ptr, c2.stride, n});
    self.builder.mut.qqq1.push_back_many({(const uint32_t *)c1.ptr, c1.stride, n});
    self.builder.mut.qqq2.push_back_many({(const uint32_t *)t.ptr, t.stride, n});
    self.builder.mut.op_types.push_back_repeat(OpType::CCX, n);
}

static inline constexpr uint8_t merge_zz_tag(QXZTypeTag8 c1, QXZTypeTag8 c2) {
    if (is_not_mixed(c1) && is_not_mixed(c2)) {
        return (uint8_t)c1 + (uint8_t)c2 * 5;
    } else {
        return 0xFF;
    }
}

static inline void append_ccx_resolve_mixed(
    PyCircuitBuilder &self, const stride_span_z &c1, const stride_span_z &c2, stride_span<const QubitId> t, size_t n) {
    for (size_t k = 0; k < n; k++) {
        self.builder.ccx(c1[k], c2[k], t[k]);
    }
}

static inline void broadcast_ccx_case_mux(
    PyCircuitBuilder &self, const stride_span_z &c2, const stride_span_z &c1, const stride_span<const QubitId> &t) {
    if (c1.count != c2.count || c1.count != t.size()) {
        throw std::invalid_argument("Incompatible lengths.");
    }
    size_t n = c1.count;
    switch (merge_zz_tag(c1.common_type, c2.common_type)) {
        case merge_zz_tag(QXZTypeTag8::BOOL_VAL, QXZTypeTag8::BOOL_VAL):
            broadcast_ccx_case(self, (stride_ptr<const bool>)c1, (stride_ptr<const bool>)c2, t, n);
            break;
        case merge_zz_tag(QXZTypeTag8::BOOL_VAL, QXZTypeTag8::BIT_ID):
            broadcast_ccx_case(self, (stride_ptr<const bool>)c1, c2.cast_data<BitId>(), t, n);
            break;
        case merge_zz_tag(QXZTypeTag8::BOOL_VAL, QXZTypeTag8::QUBIT_ID):
            broadcast_ccx_case(self, (stride_ptr<const bool>)c1, c2.cast_data<QubitId>(), t, n);
            break;
        case merge_zz_tag(QXZTypeTag8::BIT_ID, QXZTypeTag8::BOOL_VAL):
            broadcast_ccx_case(self, (stride_ptr<const bool>)c2, c1.cast_data<BitId>(), t, n);
            break;
        case merge_zz_tag(QXZTypeTag8::BIT_ID, QXZTypeTag8::BIT_ID):
            broadcast_ccx_case(self, c1.cast_data<BitId>(), c2.cast_data<BitId>(), t, n);
            break;
        case merge_zz_tag(QXZTypeTag8::BIT_ID, QXZTypeTag8::QUBIT_ID):
            broadcast_ccx_case(self, c1.cast_data<BitId>(), c2.cast_data<QubitId>(), t, n);
            break;
        case merge_zz_tag(QXZTypeTag8::QUBIT_ID, QXZTypeTag8::BOOL_VAL):
            broadcast_ccx_case(self, (stride_span<const bool>)c2, c1.cast_data<QubitId>(), t, n);
            break;
        case merge_zz_tag(QXZTypeTag8::QUBIT_ID, QXZTypeTag8::BIT_ID):
            broadcast_ccx_case(self, c2.cast_data<BitId>(), c1.cast_data<QubitId>(), t, n);
            break;
        case merge_zz_tag(QXZTypeTag8::QUBIT_ID, QXZTypeTag8::QUBIT_ID):
            broadcast_ccx_case(self, c2.cast_data<QubitId>(), c1.cast_data<QubitId>(), t, n);
            break;
        default: {
            append_ccx_resolve_mixed(self, c2, c1, t, c1.count);
        }
    }
}

void kickmix_py::broadcast_ccx_obj(
    PyCircuitBuilder &self,
    const pybind11::object &control2_obj,
    const pybind11::object &control1_obj,
    const pybind11::object &target_obj) {
    BroadcastResolver<3> resolver(
        {control2_obj, control1_obj, target_obj}, "broadcast_ccx", {"control2", "control1", "target"});
    broadcast_ccx_case_mux(
        self,
        resolver.results_xz[0].span.checked_cast_to_qubit_or_bit_or_bool("control2"),
        resolver.results_xz[1].span.checked_cast_to_qubit_or_bit_or_bool("control1"),
        resolver.results_xz[2].span.checked_cast_to_qubit_ids("target"));
}

void kickmix_py::broadcast_init_and_obj(
    PyCircuitBuilder &self,
    const pybind11::object &control1_obj,
    const pybind11::object &control2_obj,
    const pybind11::object &target_obj) {
    BroadcastResolver<3> resolver(
        {control1_obj, control2_obj, target_obj}, "broadcast_init_and", {"control1", "control2", "target"});
    stride_span_z c1 = resolver.results_xz[0].span.checked_cast_to_qubit_or_bit_or_bool("control1");
    stride_span_z c2 = resolver.results_xz[1].span.checked_cast_to_qubit_or_bit_or_bool("control2");
    stride_span<const QubitId> target = resolver.results_xz[2].span.checked_cast_to_qubit_ids("target");
    for (size_t k = 0; k < target.size(); k++) {
        self.builder.reset_and(c1[k], c2[k], target[k]);
    }
}

void kickmix_py::broadcast_del_and_obj(
    PyCircuitBuilder &self,
    const pybind11::object &control1_obj,
    const pybind11::object &control2_obj,
    const pybind11::object &target_obj) {
    BroadcastResolver<3> resolver(
        {control1_obj, control2_obj, target_obj}, "broadcast_del_and", {"control1", "control2", "target"});
    stride_span_z c1 = resolver.results_xz[0].span.checked_cast_to_qubit_or_bit_or_bool("control1");
    stride_span_z c2 = resolver.results_xz[1].span.checked_cast_to_qubit_or_bit_or_bool("control2");
    stride_span<const QubitId> target = resolver.results_xz[2].span.checked_cast_to_qubit_ids("target");
    for (size_t k = 0; k < target.size(); k++) {
        self.builder.del_and(c1[k], c2[k], target[k]);
    }
}
