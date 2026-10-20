#include "ccz.pybind.h"

#include <utility>

#include "kickmix/py/val/broadcast_resolver.pybind.h"

using namespace kickmix;
using namespace kickmix_py;

static inline void broadcast_ccz_case(
    PyCircuitBuilder &self, stride_ptr<const bool> c1, stride_ptr<const bool> c2, stride_ptr<const bool> c3, size_t n) {
    size_t total = 0;
    for (size_t k = 0; k < n; k++) {
        bool b1 = *c1++;
        bool b2 = *c2++;
        bool b3 = *c3++;
        total += b1 && b2 && b3;
    }
    self.builder.mut.op_types.push_back_repeat(OpType::NEG, total);
}

static inline void broadcast_ccz_case(
    PyCircuitBuilder &self,
    stride_ptr<const bool> c1,
    stride_ptr<const bool> c2,
    stride_ptr<const BitId> c3,
    size_t n) {
    BitId *cond_out = (BitId *)self.builder.mut.bc.grab_writeable(n);
    BitId *cond_out_start = cond_out;
    for (size_t k = 0; k < n; k++) {
        *cond_out = *c3++;
        bool b1 = *c1++;
        bool b2 = *c2++;
        bool keep = b1 && b2;
        cond_out += keep;
    }
    self.builder.mut.bc.rewind_tail((uint32_t *)cond_out);
    self.builder.mut.op_types.push_back_repeat(OpType::NEG_IF, cond_out - cond_out_start);
}

static inline void broadcast_ccz_case(
    PyCircuitBuilder &self,
    stride_ptr<const bool> c1,
    stride_ptr<const bool> c2,
    stride_ptr<const QubitId> c3,
    size_t n) {
    QubitId *q_out = (QubitId *)self.builder.mut.q0.grab_writeable(n);
    QubitId *q_out_start = q_out;
    for (size_t k = 0; k < n; k++) {
        *q_out = *c3++;
        bool b1 = *c1++;
        bool b2 = *c2++;
        bool keep = b1 && b2;
        q_out += keep;
    }
    self.builder.mut.q0.rewind_tail((uint32_t *)q_out);
    self.builder.mut.op_types.push_back_repeat(OpType::Z, q_out - q_out_start);
}

static inline void broadcast_ccz_case(
    PyCircuitBuilder &self,
    stride_ptr<const bool> c1,
    stride_ptr<const BitId> c2,
    stride_ptr<const BitId> c3,
    size_t n) {
    for (size_t k = 0; k < n; k++) {
        BitId b2 = *c2++;
        BitId b3 = *c3++;
        if (*c1++) {
            self.builder.push_condition(b2);
            self.builder.neg_if(b3);
            self.builder.pop_condition();
        }
    }
}

static inline void broadcast_ccz_case(
    PyCircuitBuilder &self,
    stride_ptr<const bool> c1,
    stride_ptr<const BitId> c2,
    stride_ptr<const QubitId> c3,
    size_t n) {
    QubitId *q_out = (QubitId *)self.builder.mut.q0.grab_writeable(n);
    BitId *cond_out = (BitId *)self.builder.mut.bc.grab_writeable(n);
    QubitId *q_out_start = q_out;
    for (size_t k = 0; k < n; k++) {
        *q_out = *c3++;
        *cond_out = *c2++;
        bool keep = *c1++ != 0;
        q_out += keep;
        cond_out += keep;
    }
    self.builder.mut.q0.rewind_tail((uint32_t *)q_out);
    self.builder.mut.bc.rewind_tail((uint32_t *)cond_out);
    self.builder.mut.op_types.push_back_repeat(OpType::Z_IF, q_out - q_out_start);
}

static inline void broadcast_ccz_case(
    PyCircuitBuilder &self,
    stride_ptr<const bool> c1,
    stride_ptr<const QubitId> c2,
    stride_ptr<const QubitId> c3,
    size_t n) {
    QubitId *q1_out = (QubitId *)self.builder.mut.qq0.grab_writeable(n);
    QubitId *q2_out = (QubitId *)self.builder.mut.qq1.grab_writeable(n);
    QubitId *q1_out_start = q1_out;
    for (size_t k = 0; k < n; k++) {
        *q1_out = *c2++;
        *q2_out = *c3++;
        bool keep = *c1++ != 0;
        q1_out += keep;
        q2_out += keep;
    }
    self.builder.mut.qq0.rewind_tail((uint32_t *)q1_out);
    self.builder.mut.qq1.rewind_tail((uint32_t *)q2_out);
    self.builder.mut.op_types.push_back_repeat(OpType::CZ, q1_out - q1_out_start);
}

static inline void broadcast_ccz_case(
    PyCircuitBuilder &self,
    stride_ptr<const BitId> c1,
    stride_ptr<const BitId> c2,
    stride_ptr<const BitId> c3,
    size_t n) {
    self.builder.pattern.pattern_push_condition(c1);
    self.builder.pattern.pattern_push_condition(c2);
    self.builder.pattern.pattern_neg_if(c3);
    self.builder.pattern.pattern_pop_condition();
    self.builder.pattern.pattern_pop_condition();
    self.builder.pattern.dump_into(self.builder.mut, 0, n, false);
}

static inline void broadcast_ccz_case(
    PyCircuitBuilder &self,
    stride_ptr<const BitId> c1,
    stride_ptr<const BitId> c2,
    stride_ptr<const QubitId> c3,
    size_t n) {
    self.builder.pattern.pattern_push_condition(c1);
    self.builder.pattern.pattern_z_if(c3, c2);
    self.builder.pattern.pattern_pop_condition();
    self.builder.pattern.dump_into(self.builder.mut, 0, n, false);
}

static inline void broadcast_ccz_case(
    PyCircuitBuilder &self,
    stride_ptr<const BitId> c1,
    stride_ptr<const QubitId> c2,
    stride_ptr<const QubitId> c3,
    size_t n) {
    self.builder.pattern.pattern_cz_if(c2, c3, c1);
    self.builder.pattern.dump_into(self.builder.mut, 0, n, false);
}

static inline void broadcast_ccz_case(
    PyCircuitBuilder &self,
    stride_ptr<const QubitId> c1,
    stride_ptr<const QubitId> c2,
    stride_ptr<const QubitId> c3,
    size_t n) {
    self.builder.mut.qqq0.push_back_many({(const uint32_t *)c1.ptr, c1.stride, n});
    self.builder.mut.qqq1.push_back_many({(const uint32_t *)c2.ptr, c2.stride, n});
    self.builder.mut.qqq2.push_back_many({(const uint32_t *)c3.ptr, c3.stride, n});
    self.builder.mut.op_types.push_back_repeat(OpType::CCZ, n);
}

static inline constexpr uint8_t merge_zzz_tag(QXZTypeTag8 c1, QXZTypeTag8 c2, QXZTypeTag8 c3) {
    if (is_not_mixed(c1) && is_not_mixed(c2) && is_not_mixed(c3)) {
        return (uint8_t)c1 + (uint8_t)c2 * 5 + (uint8_t)c3 * 25;
    } else {
        return 0xFF;
    }
}

static inline void append_ccz_resolve_mixed(
    PyCircuitBuilder &self, const stride_span_z &c1, const stride_span_z &c2, const stride_span_z &c3, size_t n) {
    for (size_t k = 0; k < n; k++) {
        self.builder.ccz(c1[k], c2[k], c3[k]);
    }
}

static inline void broadcast_ccz_case_mux(
    PyCircuitBuilder &self, stride_span_z c1, stride_span_z c2, stride_span_z c3) {
    if (c1.count != c2.count || c1.count != c3.count) {
        throw std::invalid_argument("Incompatible lengths.");
    }
    size_t n = c1.count;
    if (is_not_mixed(c1.common_type) && is_not_mixed(c2.common_type) && is_not_mixed(c3.common_type)) {
        if (c1.common_type > c2.common_type) {
            std::swap(c1, c2);
        }
        if (c2.common_type > c3.common_type) {
            std::swap(c2, c3);
        }
        if (c1.common_type > c2.common_type) {
            std::swap(c1, c2);
        }
    }
    switch (merge_zzz_tag(c1.common_type, c2.common_type, c3.common_type)) {
        case merge_zzz_tag(QXZTypeTag8::BOOL_VAL, QXZTypeTag8::BOOL_VAL, QXZTypeTag8::BOOL_VAL):
            broadcast_ccz_case(
                self, (stride_ptr<const bool>)c1, (stride_ptr<const bool>)c2, (stride_ptr<const bool>)c3, n);
            break;
        case merge_zzz_tag(QXZTypeTag8::BOOL_VAL, QXZTypeTag8::BOOL_VAL, QXZTypeTag8::BIT_ID):
            broadcast_ccz_case(self, (stride_ptr<const bool>)c1, (stride_ptr<const bool>)c2, c3.cast_data<BitId>(), n);
            break;
        case merge_zzz_tag(QXZTypeTag8::BOOL_VAL, QXZTypeTag8::BOOL_VAL, QXZTypeTag8::QUBIT_ID):
            broadcast_ccz_case(
                self, (stride_ptr<const bool>)c1, (stride_ptr<const bool>)c2, c3.cast_data<QubitId>(), n);
            break;
        case merge_zzz_tag(QXZTypeTag8::BOOL_VAL, QXZTypeTag8::BIT_ID, QXZTypeTag8::BIT_ID):
            broadcast_ccz_case(self, (stride_ptr<const bool>)c1, c2.cast_data<BitId>(), c3.cast_data<BitId>(), n);
            break;
        case merge_zzz_tag(QXZTypeTag8::BOOL_VAL, QXZTypeTag8::BIT_ID, QXZTypeTag8::QUBIT_ID):
            broadcast_ccz_case(self, (stride_ptr<const bool>)c1, c2.cast_data<BitId>(), c3.cast_data<QubitId>(), n);
            break;
        case merge_zzz_tag(QXZTypeTag8::BOOL_VAL, QXZTypeTag8::QUBIT_ID, QXZTypeTag8::QUBIT_ID):
            broadcast_ccz_case(self, (stride_ptr<const bool>)c1, c2.cast_data<QubitId>(), c3.cast_data<QubitId>(), n);
            break;
        case merge_zzz_tag(QXZTypeTag8::BIT_ID, QXZTypeTag8::BIT_ID, QXZTypeTag8::BIT_ID):
            broadcast_ccz_case(self, c1.cast_data<BitId>(), c2.cast_data<BitId>(), c3.cast_data<BitId>(), n);
            break;
        case merge_zzz_tag(QXZTypeTag8::BIT_ID, QXZTypeTag8::BIT_ID, QXZTypeTag8::QUBIT_ID):
            broadcast_ccz_case(self, c1.cast_data<BitId>(), c2.cast_data<BitId>(), c3.cast_data<QubitId>(), n);
            break;
        case merge_zzz_tag(QXZTypeTag8::BIT_ID, QXZTypeTag8::QUBIT_ID, QXZTypeTag8::QUBIT_ID):
            broadcast_ccz_case(self, c1.cast_data<BitId>(), c2.cast_data<QubitId>(), c3.cast_data<QubitId>(), n);
            break;
        case merge_zzz_tag(QXZTypeTag8::QUBIT_ID, QXZTypeTag8::QUBIT_ID, QXZTypeTag8::QUBIT_ID):
            broadcast_ccz_case(self, c1.cast_data<QubitId>(), c2.cast_data<QubitId>(), c3.cast_data<QubitId>(), n);
            break;
        default: {
            append_ccz_resolve_mixed(self, c1, c2, c3, n);
        }
    }
}

void kickmix_py::broadcast_ccz_obj(
    PyCircuitBuilder &self,
    const pybind11::object &control1_obj,
    const pybind11::object &control2_obj,
    const pybind11::object &control3_obj) {
    BroadcastResolver<3> resolver(
        {control1_obj, control2_obj, control3_obj}, "broadcast_ccz", {"control1", "control2", "control3"});
    broadcast_ccz_case_mux(
        self,
        resolver.results_xz[0].span.checked_cast_to_qubit_or_bit_or_bool("control1"),
        resolver.results_xz[1].span.checked_cast_to_qubit_or_bit_or_bool("control2"),
        resolver.results_xz[2].span.checked_cast_to_qubit_or_bit_or_bool("control3"));
}
