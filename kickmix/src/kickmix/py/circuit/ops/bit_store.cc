#include "kickmix/build/circuit_builder.h"

using namespace kickmix;

void CircuitBuilder::broadcast_bit_store(
    const stride_span<const BitId> &targets, bool value, const stride_span<const BitId> &controls) {
    if (targets.size() != controls.size()) {
        throw std::invalid_argument("broadcast_bit_store: len(targets) != len(controls)");
    }
    size_t n = targets.size();
    mut.bc.push_back_many(controls.cast_data<const uint32_t>());
    mut.b0.push_back_many(targets.cast_data<const uint32_t>());
    mut.op_types.push_back_repeat(value ? OpType::BIT_STORE1_IF : OpType::BIT_STORE0_IF, n);
}

void CircuitBuilder::broadcast_bit_store(
    const stride_span<const BitId> &targets, bool value, const stride_span<const bool> &controls) {
    if (targets.size() != controls.size()) {
        throw std::invalid_argument("broadcast_bit_store: len(targets) != len(controls)");
    }
    size_t n = targets.size();
    if (n == 0) {
        return;
    }
    OpType op = value ? OpType::BIT_STORE1 : OpType::BIT_STORE0;
    if (controls.stride == 0) {
        if (controls[0]) {
            mut.b0.push_back_many(targets.cast_data<const uint32_t>());
            mut.op_types.push_back_repeat(op, n);
        }
        return;
    }
    uint32_t *out = mut.b0.grab_writeable(n);
    uint32_t *out_start = out;
    for (size_t k = 0; k < n; k++) {
        *out = targets[k].tagged_id;
        out += controls[k];
    }
    mut.b0.rewind_tail(out);
    mut.op_types.push_back_repeat(op, out - out_start);
}

void CircuitBuilder::broadcast_bit_store(
    const stride_span<const BitId> &targets, bool value, const stride_span_z &controls) {
    if (controls.common_type == QXZTypeTag8::BIT_ID) {
        broadcast_bit_store(targets, value, controls.cast_data<BitId>());
        return;
    }
    if (controls.common_type == QXZTypeTag8::BOOL_VAL) {
        broadcast_bit_store(targets, value, (stride_span<const bool>)controls);
        return;
    }
    if (targets.size() != controls.size()) {
        throw std::invalid_argument("broadcast_bit_store: len(targets) != len(controls)");
    }
    if (!controls.is_all_classical()) {
        throw std::invalid_argument("bit_store: control must be classical (a bit or a bool), not a qubit.");
    }
    size_t n = targets.size();
    for (size_t k = 0; k < n; k++) {
        QubitOrBitOrBool c = controls[k];
        if (c.is_bit()) {
            if (value) {
                bit_store1_if(targets[k], (BitId)c);
            } else {
                bit_store0_if(targets[k], (BitId)c);
            }
        } else if ((bool)c) {
            if (value) {
                bit_store1(targets[k]);
            } else {
                bit_store0(targets[k]);
            }
        }
    }
}
