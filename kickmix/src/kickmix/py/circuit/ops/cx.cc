#include "kickmix/build/circuit_builder.h"

using namespace kickmix;

void CircuitBuilder::broadcast_cx(const stride_span<const bool> &controls, const stride_span<const QubitId> &targets) {
    if (controls.size() != targets.size()) {
        throw std::invalid_argument("broadcast_cx: len(controls) != len(targets)");
    }
    size_t n = targets.size();

    uint32_t *out = mut.q0.grab_writeable(n);
    uint32_t *out_start = out;
    for (size_t k = 0; k < n; k++) {
        *out = targets[k].tagged_id;
        out += controls[k];
    }
    mut.q0.rewind_tail(out);
    mut.op_types.push_back_repeat(OpType::X, out - out_start);
}

void CircuitBuilder::broadcast_cx(const stride_span<const BitId> &controls, const stride_span<const QubitId> &targets) {
    if (controls.size() != targets.size()) {
        throw std::invalid_argument("broadcast_cx: len(controls) != len(targets)");
    }
    size_t n = targets.size();
    mut.bc.push_back_many(controls.cast_data<uint32_t>());
    mut.q0.push_back_many(targets.cast_data<uint32_t>());
    mut.op_types.push_back_repeat(OpType::X_IF, n);
}

void CircuitBuilder::broadcast_cx(
    const stride_span<const QubitId> &controls, const stride_span<const QubitId> &targets) {
    if (controls.size() != targets.size()) {
        throw std::invalid_argument("broadcast_cx: len(controls) != len(targets)");
    }
    size_t n = targets.size();
    mut.qq0.push_back_many(controls.cast_data<uint32_t>());
    mut.qq1.push_back_many(targets.cast_data<uint32_t>());
    mut.op_types.push_back_repeat(OpType::CX, n);
}

void CircuitBuilder::broadcast_cx(
    const stride_span<const QubitId> &controls, const stride_span<const XBitId> &targets) {
    if (controls.size() != targets.size()) {
        throw std::invalid_argument("broadcast_cx: len(controls) != len(targets)");
    }
    size_t n = targets.size();
    mut.q0.push_back_many(controls.cast_data<uint32_t>());
    mut.bc.push_back_many(targets.cast_data<uint32_t>());
    mut.op_types.push_back_repeat(OpType::Z_IF, n);
}

void CircuitBuilder::broadcast_cx(const stride_span<const BitId> &controls, const stride_span<const XBitId> &targets) {
    if (controls.size() != targets.size()) {
        throw std::invalid_argument("broadcast_cx: len(controls) != len(targets)");
    }
    size_t n = targets.size();
    pattern.pattern_push_condition(targets.cast_data<const BitId>());
    pattern.pattern_neg_if(controls);
    pattern.pattern_pop_condition();
    pattern.dump_into(mut, 0, n, false);
}

void CircuitBuilder::broadcast_cx(const stride_span<const bool> &controls, const stride_span<const XBitId> &targets) {
    if (controls.size() != targets.size()) {
        throw std::invalid_argument("broadcast_cx: len(controls) != len(targets)");
    }
    size_t n = targets.size();
    uint32_t *out = mut.bc.grab_writeable(n);
    uint32_t *out_start = out;
    for (size_t k = 0; k < n; k++) {
        *out = targets[k].tagged_id;
        out += controls[k];
    }
    mut.bc.rewind_tail(out);
    mut.op_types.push_back_repeat(OpType::NEG_IF, out - out_start);
}

void CircuitBuilder::broadcast_cx(const stride_span<const QubitId> &controls, const stride_span<const XBool> &targets) {
    if (controls.size() != targets.size()) {
        throw std::invalid_argument("broadcast_cx: len(controls) != len(targets)");
    }
    size_t n = targets.size();
    uint32_t *out = mut.q0.grab_writeable(n);
    uint32_t *out_start = out;
    for (size_t k = 0; k < n; k++) {
        *out = controls[k].tagged_id;
        out += targets[k].is_minus_ket();
    }
    mut.q0.rewind_tail(out);
    mut.op_types.push_back_repeat(OpType::Z, out - out_start);
}

void CircuitBuilder::broadcast_cx(const stride_span<const BitId> &controls, const stride_span<const XBool> &targets) {
    if (controls.size() != targets.size()) {
        throw std::invalid_argument("broadcast_cx: len(controls) != len(targets)");
    }
    size_t n = targets.size();
    uint32_t *out = mut.bc.grab_writeable(n);
    uint32_t *out_start = out;
    for (size_t k = 0; k < n; k++) {
        *out = controls[k].tagged_id;
        out += targets[k].is_minus_ket();
    }
    mut.bc.rewind_tail(out);
    mut.op_types.push_back_repeat(OpType::NEG_IF, out - out_start);
}

void CircuitBuilder::broadcast_cx(const stride_span<const bool> &controls, const stride_span<const XBool> &targets) {
    if (controls.size() != targets.size()) {
        throw std::invalid_argument("broadcast_cx: len(controls) != len(targets)");
    }
    size_t n = targets.size();
    size_t total = 0;
    for (size_t k = 0; k < n; k++) {
        total += controls[k] && targets[k].is_minus_ket();
    }
    mut.op_types.push_back_repeat(OpType::NEG, total);
}

void CircuitBuilder::broadcast_cx(const stride_span<const BitId> &controls, const stride_span<const BitId> &targets) {
    if (controls.size() != targets.size()) {
        throw std::invalid_argument("broadcast_cx: len(controls) != len(targets)");
    }
    size_t n = targets.size();
    mut.bc.push_back_many(controls.cast_data<uint32_t>());
    mut.b0.push_back_many(targets.cast_data<uint32_t>());
    mut.op_types.push_back_repeat(OpType::BIT_INVERT_IF, n);
}

void CircuitBuilder::broadcast_cx(const stride_span<const bool> &controls, const stride_span<const BitId> &targets) {
    if (controls.size() != targets.size()) {
        throw std::invalid_argument("broadcast_cx: len(controls) != len(targets)");
    }
    size_t n = targets.size();
    uint32_t *out = mut.bc.grab_writeable(n);
    uint32_t *out_start = out;
    for (size_t k = 0; k < n; k++) {
        *out = targets[k].tagged_id;
        out += controls[k];
    }
    mut.bc.rewind_tail(out);
    mut.op_types.push_back_repeat(OpType::BIT_INVERT, out - out_start);
}

void CircuitBuilder::mux_cx(
    const stride_span<const QubitOrXZBitOrXZBool> &controls, const stride_span<const QubitOrXZBitOrXZBool> &targets) {
    if (controls.size() != targets.size()) {
        throw std::invalid_argument("broadcast_cx: len(controls) != len(targets)");
    }
    size_t n = targets.size();
    for (size_t k = 0; k < n; k++) {
        mux_cx(controls[k], targets[k]);
    }
}

void CircuitBuilder::mux_broadcast_cx(
    const stride_span<const QubitOrXZBitOrXZBool> &controls, const stride_span<const QubitOrXZBitOrXZBool> &targets) {
    if (controls.size() != targets.size()) {
        throw std::invalid_argument("broadcast_cx: len(controls) != len(targets)");
    }
    size_t n = targets.size();
    for (size_t k = 0; k < n; k++) {
        mux_cx(controls[k], targets[k]);
    }
}

void CircuitBuilder::broadcast_cx(const stride_span_z &controls, const stride_span_x &targets) {
    mux_broadcast_cx(controls, targets);
}
void CircuitBuilder::mux_broadcast_cx(const stride_span_xz &controls, const stride_span_xz &targets) {
    if (targets.common_type == QXZTypeTag8::QUBIT_ID) {
        if (controls.common_type == QXZTypeTag8::BOOL_VAL) {
            broadcast_cx(controls.cast_data<bool>(), targets.cast_data<QubitId>());
            return;
        } else if (controls.common_type == QXZTypeTag8::BIT_ID) {
            broadcast_cx(controls.cast_data<BitId>(), targets.cast_data<QubitId>());
            return;
        } else if (controls.common_type == QXZTypeTag8::QUBIT_ID) {
            broadcast_cx(controls.cast_data<QubitId>(), targets.cast_data<QubitId>());
            return;
        }
    } else if (targets.common_type == QXZTypeTag8::XBIT_ID) {
        if (controls.common_type == QXZTypeTag8::BOOL_VAL) {
            broadcast_cx(controls.cast_data<bool>(), targets.cast_data<XBitId>());
            return;
        } else if (controls.common_type == QXZTypeTag8::BIT_ID) {
            broadcast_cx(controls.cast_data<BitId>(), targets.cast_data<XBitId>());
            return;
        } else if (controls.common_type == QXZTypeTag8::QUBIT_ID) {
            broadcast_cx(controls.cast_data<QubitId>(), targets.cast_data<XBitId>());
            return;
        }
    } else if (targets.common_type == QXZTypeTag8::XBOOL_VAL) {
        if (controls.common_type == QXZTypeTag8::BOOL_VAL) {
            broadcast_cx(controls.cast_data<bool>(), targets.cast_data<XBool>());
            return;
        } else if (controls.common_type == QXZTypeTag8::BIT_ID) {
            broadcast_cx(controls.cast_data<BitId>(), targets.cast_data<XBool>());
            return;
        } else if (controls.common_type == QXZTypeTag8::QUBIT_ID) {
            broadcast_cx(controls.cast_data<QubitId>(), targets.cast_data<XBool>());
            return;
        }
    } else if (targets.common_type == QXZTypeTag8::BIT_ID) {
        if (controls.common_type == QXZTypeTag8::BIT_ID) {
            broadcast_cx(controls.cast_data<BitId>(), targets.cast_data<BitId>());
            return;
        } else if (controls.common_type == QXZTypeTag8::BOOL_VAL) {
            broadcast_cx(controls.cast_data<bool>(), targets.cast_data<BitId>());
            return;
        }
    }

    mux_broadcast_cx(controls.cast_data<QubitOrXZBitOrXZBool>(), targets.cast_data<QubitOrXZBitOrXZBool>());
}

void CircuitBuilder::mux_cx(QubitOrXZBitOrXZBool control, QubitOrXZBitOrXZBool target) {
    if (control.is_qubit_or_bit_or_bool() && target.is_qubit_or_xbit_or_xbool()) {
        cx((QubitOrBitOrBool)control, (QubitOrXBitOrXBool)target);
    }
}

void CircuitBuilder::cx(bool control, BitId target) {
    if (control) {
        bit_invert(target);
    }
}

void CircuitBuilder::cx(BitOrBool control, BitId target) {
    if (control.is_bit()) {
        bit_invert_if(target, (BitId)control);
    } else if ((bool)control) {
        bit_invert(target);
    }
}

void CircuitBuilder::cx(BitId control, BitId target) {
    bit_invert_if(target, control);
}

void CircuitBuilder::cx(QubitId control, QubitId target) {
    mut.op_types.push_back(OpType::CX);
    mut.qq0.push_back(control.tagged_id);
    mut.qq1.push_back(target.tagged_id);
}

void CircuitBuilder::cx(QubitOrBitOrBool control, QubitOrXBitOrXBool target) {
    if (control.is_qubit()) {
        if (target.is_qubit()) {
            cx((QubitId)control, (QubitId)target);
        } else if (target.is_xbit()) {
            z_if((QubitId)control, ((XBitId)target).conjugated_by_h());
        } else if (target.is_minus_ket()) {
            z((QubitId)control);
        }
    } else if (control.is_bit()) {
        if (target.is_qubit()) {
            x_if((QubitId)target, (BitId)control);
        } else if (target.is_xbit()) {
            push_condition(((XBitId)target).conjugated_by_h());
            neg_if((BitId)control);
            pop_condition();
        } else if (target.is_minus_ket()) {
            neg_if((BitId)control);
        }
    } else if ((bool)control) {
        x(target);
    }
}

void CircuitBuilder::broadcast_cx(QubitId control, const stride_span<const QubitId> &targets) {
    size_t n = targets.size();
    mut.qq0.push_back_repeat(control.tagged_id, n);
    mut.qq1.push_back_many(targets.cast_data<uint32_t>());
    mut.op_types.push_back_repeat(OpType::CX, n);
}

void CircuitBuilder::broadcast_cx(BitId control, const stride_span<const QubitId> &targets) {
    push_condition(control);
    broadcast_x(targets);
    pop_condition();
}