#ifndef KICKMIX_CIRCUIT_PATTERN_H
#define KICKMIX_CIRCUIT_PATTERN_H

#include <vector>

#include "kickmix/circuit/circuit.h"
#include "kickmix/circuit/mutable_circuit.h"
#include "kickmix/circuit/op_type.h"
#include "kickmix/id/array_z.h"
#include "kickmix/mem/monotonic_arena.h"
#include "kickmix/mem/stride_span.h"

namespace kickmix {

template <typename T>
struct spanish {
    bool has_stride;
    union {
        stride_ptr<const T> stride;
        T val;
    };
    spanish(T val) : has_stride(false), val(val) {
    }
    spanish(stride_ptr<const T> val) : has_stride(true), stride(val) {
    }
    spanish(stride_span<const T> val) : has_stride(true), stride(val) {
    }
};

struct CircuitPattern {
    MonotonicArena<uint32_t, 32> tmp_data;
    std::vector<OpType> pat_op_types;
    std::vector<stride_ptr<const uint32_t>> pat_qqq0;
    std::vector<stride_ptr<const uint32_t>> pat_qqq1;
    std::vector<stride_ptr<const uint32_t>> pat_qqq2;
    std::vector<stride_ptr<const uint32_t>> pat_qq0;
    std::vector<stride_ptr<const uint32_t>> pat_qq1;
    std::vector<stride_ptr<const uint32_t>> pat_q0;
    std::vector<stride_ptr<const uint32_t>> pat_bc;
    std::vector<stride_ptr<const uint32_t>> pat_b0;

    CircuitPattern() = default;
    CircuitPattern(CircuitPattern &&) noexcept = default;
    CircuitPattern &operator=(CircuitPattern &&) = default;

    template <typename T>
    stride_ptr<const uint32_t> ptr(spanish<const T> s) {
        if (s.has_stride) {
            return stride_ptr<const uint32_t>{(uint32_t *)s.stride.ptr, s.stride.stride};
        } else {
            uint32_t *v = tmp_data.grab_writeable(1);
            *v = s.val.tagged_id;
            return stride_ptr<const uint32_t>(v, 0);
        }
    }

    void pattern_swap(const spanish<const QubitId> &target1, const spanish<const QubitId> &target2);
    void pattern_r(const spanish<const QubitId> &target);
    void pattern_r_if(const spanish<const QubitId> &target, const spanish<const BitId> &condition);
    void pattern_hmr(const spanish<const QubitId> &target, const spanish<const BitId> &output);
    void pattern_x(const spanish<const QubitId> &target);
    void pattern_z(const spanish<const QubitId> &target);
    void pattern_bit_invert_if(const spanish<const BitId> &target, const spanish<const BitId> &condition);
    void pattern_x_if(const spanish<const QubitId> &, const spanish<const BitId> &condition);
    void pattern_cz_if(
        const spanish<const QubitId> &control,
        const spanish<const QubitId> &target,
        const spanish<const BitId> &condition);

    void mux_pattern_cz(const stride_span_z &control, const spanish<const QubitId> &target);
    void mux_pattern_z(const stride_span_z &control);
    void pattern_z_if(const spanish<const QubitId> &target, const spanish<const BitId> &condition);
    void mux_pattern_z_if(const QubitOrTrue &control, const spanish<const BitId> &condition);
    void pattern_neg_if(const spanish<const BitId> &condition);
    void pattern_cz(const spanish<const QubitId> &control, const spanish<const QubitId> &target);

    void pattern_push_condition(const spanish<const BitId> &cond);
    void pattern_pop_condition();
    void mux_pattern_cx(const stride_span_z &control, const spanish<const QubitId> &target);
    void mux_pattern_cx_if(
        const QubitOrTrue &control, const spanish<const QubitId> &target, const spanish<const BitId> &condition);
    void mux_pattern_ccx_if(
        const QubitOrTrue &control,
        const spanish<const QubitId> &control2,
        const spanish<const QubitId> &target,
        const spanish<const BitId> &condition);
    void mux_pattern_ccx(
        const QubitOrTrue &control, const spanish<const QubitId> &control2, const spanish<const QubitId> &target);
    void mux_pattern_ccz(
        const QubitOrTrue &control, const spanish<const QubitId> &control2, const spanish<const QubitId> &target);
    void pattern_cx_if(
        const spanish<const QubitId> &control,
        const spanish<const QubitId> &target,
        const spanish<const BitId> &condition);
    void pattern_cx(const spanish<const QubitId> &control, const spanish<const QubitId> &target);
    void pattern_ccx(
        const spanish<const QubitId> &control1,
        const spanish<const QubitId> &control2,
        const spanish<const QubitId> &target);
    void pattern_ccz(
        const spanish<const QubitId> &control1,
        const spanish<const QubitId> &control2,
        const spanish<const QubitId> &target);
    void pattern_ccx_if(
        const spanish<const QubitId> &control1,
        const spanish<const QubitId> &control2,
        const spanish<const QubitId> &target,
        const spanish<const BitId> &cond);
    void dump_into(MutableCircuit &mut, size_t start, size_t end, bool reversed);
};
struct LoopBuilder {
    CircuitPattern *out;

    void reset(spanish<const QubitId> p) {
        out->pattern_r(p);
    }
    void swap(spanish<const QubitId> target1, spanish<const QubitId> target2) {
        out->pattern_swap(target1, target2);
    }
    void x(spanish<const QubitId> t) {
        out->pattern_x(t);
    }
    void z(spanish<const QubitId> t) {
        out->pattern_z(t);
    }
    void neg_if(spanish<const BitId> t) {
        out->pattern_neg_if(t);
    }
    void x_if(spanish<const QubitId> t, spanish<const BitId> cond) {
        out->pattern_x_if(t, cond);
    }
    void z_if(spanish<const QubitId> t, spanish<const BitId> cond) {
        out->pattern_z_if(t, cond);
    }
    void cx(spanish<const QubitId> c, spanish<const QubitId> t) {
        out->pattern_cx(c, t);
    }
    void cz(spanish<const QubitId> c, spanish<const QubitId> t) {
        out->pattern_cz(c, t);
    }
    void cz(spanish<const QubitId> c, spanish<const BitId> t) {
        out->pattern_z_if(c, t);
    }
    void bit_invert_if(const spanish<const BitId> &target, const spanish<const BitId> &condition) {
        out->pattern_bit_invert_if(target, condition);
    }
    void mux_cx_if(
        const QubitOrTrue &control, const spanish<const QubitId> &target, const spanish<const BitId> &condition) {
        if (control.is_qubit()) {
            cx_if((QubitId)control, target, condition);
        } else {
            x_if(target, condition);
        }
    }
    void cx_if(spanish<const QubitId> control, spanish<const QubitId> target, spanish<const BitId> condition) {
        out->pattern_cx_if(control, target, condition);
    }
    void hmr(spanish<const QubitId> q, spanish<const BitId> t) {
        out->pattern_hmr(q, t);
    }
    void push_condition(spanish<const BitId> t) {
        out->pattern_push_condition(t);
    }
    void pop_condition() {
        out->pattern_pop_condition();
    }
    void cz_if(spanish<const QubitId> c1, spanish<const QubitId> c2, spanish<const BitId> cond) {
        out->pattern_cz_if(c1, c2, cond);
    }
    void ccx(spanish<const QubitId> c1, spanish<const QubitId> c2, spanish<const QubitId> t) {
        out->pattern_ccx(c1, c2, t);
    }
    void ccz(spanish<const QubitId> c1, spanish<const QubitId> c2, spanish<const QubitId> t) {
        out->pattern_ccz(c1, c2, t);
    }
    void mux_ccx(const QubitOrTrue &c1, spanish<const QubitId> c2, spanish<const QubitId> t) {
        out->mux_pattern_ccx(c1, c2, t);
    }
    void mux_z_if(const QubitOrTrue &control, const spanish<const BitId> &condition) {
        out->mux_pattern_z_if(control, condition);
    }
};

}  // namespace kickmix

#endif
