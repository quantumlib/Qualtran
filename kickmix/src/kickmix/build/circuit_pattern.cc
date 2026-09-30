#include "circuit_builder.h"
#include "kickmix/mem/util.h"

using namespace kickmix;

void CircuitPattern::pattern_swap(const spanish<const QubitId> &target1, const spanish<const QubitId> &target2) {
    pat_op_types.push_back(OpType::SWAP);
    pat_qq0.push_back(ptr(target1));
    pat_qq1.push_back(ptr(target2));
}

void CircuitPattern::pattern_r(const spanish<const QubitId> &target) {
    pat_op_types.push_back(OpType::R);
    pat_q0.push_back(ptr(target));
}

void CircuitPattern::pattern_r_if(const spanish<const QubitId> &target, const spanish<const BitId> &condition) {
    pat_op_types.push_back(OpType::R_IF);
    pat_bc.push_back(ptr(condition));
    pat_q0.push_back(ptr(target));
}

void CircuitPattern::pattern_x(const spanish<const QubitId> &target) {
    pat_op_types.push_back(OpType::X);
    pat_q0.push_back(ptr(target));
}

void CircuitPattern::pattern_z(const spanish<const QubitId> &target) {
    pat_op_types.push_back(OpType::Z);
    pat_q0.push_back(ptr(target));
}

void CircuitPattern::pattern_bit_invert_if(const spanish<const BitId> &target, const spanish<const BitId> &condition) {
    pat_op_types.push_back(OpType::BIT_INVERT_IF);
    pat_bc.push_back(ptr(condition));
    pat_b0.push_back(ptr(target));
}

void CircuitPattern::pattern_x_if(const spanish<const QubitId> &target, const spanish<const BitId> &condition) {
    pat_op_types.push_back(OpType::X_IF);
    pat_bc.push_back(ptr(condition));
    pat_q0.push_back(ptr(target));
}

void CircuitPattern::mux_pattern_cx(const stride_span_z &control, const spanish<const QubitId> &target) {
    if (control.common_type == QXZTypeTag8::QUBIT_ID) {
        pattern_cx(control.cast_data<QubitId>()[iota{}], target);
    } else if (control.common_type == QXZTypeTag8::BIT_ID) {
        pattern_x_if(target, control.cast_data<BitId>()[iota{}]);
    } else {
        throw std::invalid_argument(
            "mux_pattern_cx requires control.common_type == QXZTypeTag8::QUBIT_ID or QXZTypeTag8::BIT_ID");
    }
}

void CircuitPattern::mux_pattern_ccx_if(
    const QubitOrTrue &control,
    const spanish<const QubitId> &control2,
    const spanish<const QubitId> &target,
    const spanish<const BitId> &condition) {
    if (control.is_qubit()) {
        pattern_ccx_if((QubitId)control, control2, target, condition);
    } else {
        pattern_cx_if(control2, target, condition);
    }
}

void CircuitPattern::mux_pattern_cx_if(
    const QubitOrTrue &control, const spanish<const QubitId> &target, const spanish<const BitId> &condition) {
    if (control.is_qubit()) {
        pattern_cx_if((QubitId)control, target, condition);
    } else {
        pattern_x_if(target, condition);
    }
}

void CircuitPattern::mux_pattern_z_if(const QubitOrTrue &control, const spanish<const BitId> &condition) {
    if (control.is_qubit()) {
        pattern_z_if((QubitId)control, condition);
    } else {
        pattern_neg_if(condition);
    }
}

void CircuitPattern::mux_pattern_ccx(
    const QubitOrTrue &control, const spanish<const QubitId> &control2, const spanish<const QubitId> &target) {
    if (control.is_qubit()) {
        pattern_ccx((QubitId)control, control2, target);
    } else {
        pattern_cx(control2, target);
    }
}

void CircuitPattern::mux_pattern_ccz(
    const QubitOrTrue &control, const spanish<const QubitId> &control2, const spanish<const QubitId> &target) {
    if (control.is_qubit()) {
        pattern_ccz((QubitId)control, control2, target);
    } else {
        pattern_cz(control2, target);
    }
}

void CircuitPattern::pattern_cx_if(
    const spanish<const QubitId> &control,
    const spanish<const QubitId> &target,
    const spanish<const BitId> &condition) {
    pat_op_types.push_back(OpType::CX_IF);
    pat_qq0.push_back(ptr(control));
    pat_qq1.push_back(ptr(target));
    pat_bc.push_back(ptr(condition));
}

void CircuitPattern::pattern_cz_if(
    const spanish<const QubitId> &control,
    const spanish<const QubitId> &target,
    const spanish<const BitId> &condition) {
    pat_op_types.push_back(OpType::CZ_IF);
    pat_qq0.push_back(ptr(control));
    pat_qq1.push_back(ptr(target));
    pat_bc.push_back(ptr(condition));
}

void CircuitPattern::pattern_cx(const spanish<const QubitId> &control, const spanish<const QubitId> &target) {
    pat_op_types.push_back(OpType::CX);
    pat_qq0.push_back(ptr(control));
    pat_qq1.push_back(ptr(target));
}

void CircuitPattern::mux_pattern_cz(const stride_span_z &control, const spanish<const QubitId> &target) {
    if (control.common_type == QXZTypeTag8::QUBIT_ID) {
        pattern_cz(control.cast_data<QubitId>()[iota{}], target);
    } else if (control.common_type == QXZTypeTag8::BIT_ID) {
        pattern_z_if(target, control.cast_data<BitId>()[iota{}]);
    } else {
        throw std::invalid_argument(
            "mux_pattern_cz requires control.common_type == QXZTypeTag8::QUBIT_ID or QXZTypeTag8::BIT_ID");
    }
}
void CircuitPattern::mux_pattern_z(const stride_span_z &control) {
    if (control.common_type == QXZTypeTag8::QUBIT_ID) {
        pattern_z(control.cast_data<QubitId>()[iota{}]);
    } else if (control.common_type == QXZTypeTag8::BIT_ID) {
        pattern_neg_if(control.cast_data<BitId>()[iota{}]);
    } else {
        throw std::invalid_argument(
            "mux_pattern_z requires control.common_type == QXZTypeTag8::QUBIT_ID or QXZTypeTag8::BIT_ID");
    }
}
void CircuitPattern::pattern_z_if(const spanish<const QubitId> &target, const spanish<const BitId> &condition) {
    pat_op_types.push_back(OpType::Z_IF);
    pat_bc.push_back(ptr(condition));
    pat_q0.push_back(ptr(target));
}
void CircuitPattern::pattern_neg_if(const spanish<const BitId> &condition) {
    pat_op_types.push_back(OpType::NEG_IF);
    pat_bc.push_back(ptr(condition));
}
void CircuitPattern::pattern_cz(const spanish<const QubitId> &control, const spanish<const QubitId> &target) {
    pat_op_types.push_back(OpType::CZ);
    pat_qq0.push_back(ptr(control));
    pat_qq1.push_back(ptr(target));
}

void CircuitPattern::pattern_hmr(const spanish<const QubitId> &target, const spanish<const BitId> &output) {
    pat_op_types.push_back(OpType::HMR);
    pat_q0.push_back(ptr(target));
    pat_b0.push_back(ptr(output));
}

void CircuitPattern::pattern_ccx(
    const spanish<const QubitId> &control1,
    const spanish<const QubitId> &control2,
    const spanish<const QubitId> &target) {
    pat_op_types.push_back(OpType::CCX);
    pat_qqq0.push_back(ptr(control2));
    pat_qqq1.push_back(ptr(control1));
    pat_qqq2.push_back(ptr(target));
}

void CircuitPattern::pattern_ccz(
    const spanish<const QubitId> &control1,
    const spanish<const QubitId> &control2,
    const spanish<const QubitId> &target) {
    pat_op_types.push_back(OpType::CCZ);
    pat_qqq0.push_back(ptr(control2));
    pat_qqq1.push_back(ptr(control1));
    pat_qqq2.push_back(ptr(target));
}

void CircuitPattern::pattern_ccx_if(
    const spanish<const QubitId> &control1,
    const spanish<const QubitId> &control2,
    const spanish<const QubitId> &target,
    const spanish<const BitId> &cond) {
    pat_op_types.push_back(OpType::CCX_IF);
    pat_qqq0.push_back(ptr(control2));
    pat_qqq1.push_back(ptr(control1));
    pat_qqq2.push_back(ptr(target));
    pat_bc.push_back(ptr(cond));
}

void CircuitPattern::pattern_push_condition(const spanish<const BitId> &cond) {
    pat_op_types.push_back(OpType::PUSH_CONDITION);
    pat_bc.push_back(ptr(cond));
}

void CircuitPattern::pattern_pop_condition() {
    pat_op_types.push_back(OpType::POP_CONDITION);
}

static void gather_and_clear(
    MonotonicArena<uint32_t, 32> &buf,
    std::vector<stride_ptr<const uint32_t>> &inputs,
    size_t start,
    size_t stop,
    bool reversed) {
    size_t repetitions = stop - start;
    if (reversed) {
        for (auto &p : inputs) {
            p += stop;
            p -= 1;
            p.stride *= -1;
        }
    } else {
        for (auto &p : inputs) {
            p += start;
        }
    }

    uint32_t *out = buf.grab_writeable(inputs.size() * repetitions);

    switch (inputs.size()) {
        case 0:
            break;
        case 1: {
            const uint32_t *p0 = inputs[0].ptr;
            int64_t s0 = inputs[0].stride;
            for (size_t k = 0; k < repetitions; k++) {
                *out++ = *p0;
                p0 += s0;
            }
            break;
        }
        case 2: {
            const uint32_t *p0 = inputs[0].ptr;
            const uint32_t *p1 = inputs[1].ptr;
            int64_t s0 = inputs[0].stride;
            int64_t s1 = inputs[1].stride;
            for (size_t k = 0; k < repetitions; k++) {
                *out++ = *p0;
                *out++ = *p1;
                p0 += s0;
                p1 += s1;
            }
            break;
        }
        case 3: {
            const uint32_t *p0 = inputs[0].ptr;
            const uint32_t *p1 = inputs[1].ptr;
            const uint32_t *p2 = inputs[2].ptr;
            int64_t s0 = inputs[0].stride;
            int64_t s1 = inputs[1].stride;
            int64_t s2 = inputs[2].stride;
            for (size_t k = 0; k < repetitions; k++) {
                *out++ = *p0;
                *out++ = *p1;
                *out++ = *p2;
                p0 += s0;
                p1 += s1;
                p2 += s2;
            }
            break;
        }
        default: {
            for (size_t k = 0; k < repetitions; k++) {
                for (size_t r = 0; r < inputs.size(); r++) {
                    *out++ = *inputs[r]++;
                }
            }
        }
    }

    inputs.clear();
}

void CircuitPattern::dump_into(MutableCircuit &mut, size_t start, size_t end, bool reversed) {
    if (start > end) {
        start = end;
    }

    {
        size_t reps = end - start;
        memcpy_repeat(
            mut.op_types.grab_writeable(reps * pat_op_types.size()), pat_op_types.data(), pat_op_types.size(), reps);
        pat_op_types.clear();
    }

    if (!pat_bc.empty()) {
        gather_and_clear(mut.bc, pat_bc, start, end, reversed);
    }

    if (!pat_b0.empty()) {
        gather_and_clear(mut.b0, pat_b0, start, end, reversed);
    }

    if (!pat_q0.empty()) {
        gather_and_clear(mut.q0, pat_q0, start, end, reversed);
    }

    if (!pat_qq0.empty()) {
        gather_and_clear(mut.qq0, pat_qq0, start, end, reversed);
        gather_and_clear(mut.qq1, pat_qq1, start, end, reversed);
    }

    if (!pat_qqq0.empty()) {
        gather_and_clear(mut.qqq0, pat_qqq0, start, end, reversed);
        gather_and_clear(mut.qqq1, pat_qqq1, start, end, reversed);
        gather_and_clear(mut.qqq2, pat_qqq2, start, end, reversed);
    }

    tmp_data.clear();
}
