#include <iostream>

#include "kickmix/build/circuit_builder.h"

using namespace kickmix;

void CircuitBuilder::broadcast_cz(const stride_span<const bool> &controls1, const stride_span<const bool> &controls2) {
    if (controls1.size() != controls2.size()) {
        throw std::invalid_argument("broadcast_cz: len(controls1) != len(controls2)");
    }
    size_t n = controls1.size();
    size_t total = 0;
    for (size_t k = 0; k < n; k++) {
        bool b1 = controls1[k];
        bool b2 = controls2[k];
        total += b1 && b2;
    }
    // DIDNTDO: reduce modulo 2 (because it would break equivalence with the naive loop).
    mut.op_types.push_back_repeat(OpType::NEG, total);
}

void CircuitBuilder::broadcast_cz(
    const stride_span<const bool> &controls1, const stride_span<const QubitId> &controls2) {
    if (controls1.size() != controls2.size()) {
        throw std::invalid_argument("broadcast_cz: len(controls1) != len(controls2)");
    }
    size_t n = controls1.size();

    uint32_t *out = mut.q0.grab_writeable(n);
    uint32_t *out_start = out;
    for (size_t k = 0; k < n; k++) {
        *out = controls2[k].tagged_id;
        bool keep = controls1[k];
        out += keep;
    }
    mut.q0.rewind_tail(out);
    mut.op_types.push_back_repeat(OpType::Z, out - out_start);
}

void CircuitBuilder::broadcast_cz(const stride_span<const bool> &controls1, const stride_span<const BitId> &controls2) {
    if (controls1.size() != controls2.size()) {
        throw std::invalid_argument("broadcast_cz: len(controls1) != len(controls2)");
    }
    size_t n = controls1.size();
    uint32_t *out = mut.bc.grab_writeable(n);
    uint32_t *out_start = out;
    for (size_t k = 0; k < n; k++) {
        *out = controls2[k].tagged_id;
        bool keep = controls1[k] != 0;
        out += keep;
    }
    mut.bc.rewind_tail(out);
    mut.op_types.push_back_repeat(OpType::NEG_IF, out - out_start);
}

void CircuitBuilder::broadcast_cz(
    const stride_span<const BitId> &controls1, const stride_span<const BitId> &controls2) {
    if (controls1.size() != controls2.size()) {
        throw std::invalid_argument("broadcast_cz: len(controls1) != len(controls2)");
    }
    size_t n = controls1.size();
    pattern.pattern_push_condition(controls2);
    pattern.pattern_neg_if(controls1);
    pattern.pattern_pop_condition();
    pattern.dump_into(mut, 0, n, false);
}

void CircuitBuilder::broadcast_cz(
    const stride_span<const BitId> &controls1, const stride_span<const QubitId> &controls2) {
    if (controls1.size() != controls2.size()) {
        throw std::invalid_argument("broadcast_cz: len(controls1) != len(controls2)");
    }
    size_t n = controls1.size();
    mut.q0.push_back_many(controls2.cast_data<const uint32_t>());
    mut.bc.push_back_many(controls1.cast_data<const uint32_t>());
    mut.op_types.push_back_repeat(OpType::Z_IF, n);
}

void CircuitBuilder::broadcast_cz(
    const stride_span<const QubitId> &controls1, const stride_span<const QubitId> &controls2) {
    if (controls1.size() != controls2.size()) {
        throw std::invalid_argument("broadcast_cz: len(controls1) != len(controls2)");
    }
    size_t n = controls1.size();
    mut.qq0.push_back_many(controls1.cast_data<const uint32_t>());
    mut.qq1.push_back_many(controls2.cast_data<const uint32_t>());
    mut.op_types.push_back_repeat(OpType::CZ, n);
}

void CircuitBuilder::broadcast_cz(
    const stride_span<const QubitOrBitOrBool> &controls1, const stride_span<const QubitOrBitOrBool> &controls2) {
    if (controls1.size() != controls2.size()) {
        throw std::invalid_argument("broadcast_cz: len(controls1) != len(controls2)");
    }
    size_t n = controls1.size();
    for (size_t k = 0; k < n; k++) {
        cz(controls1[k], controls2[k]);
    }
}

void CircuitBuilder::mux_broadcast_cz(
    const stride_span<const QubitOrXZBitOrXZBool> &controls1,
    const stride_span<const QubitOrXZBitOrXZBool> &controls2) {
    if (controls1.size() != controls2.size()) {
        throw std::invalid_argument("broadcast_cz: len(controls1) != len(controls2)");
    }
    size_t n = controls1.size();
    for (size_t k = 0; k < n; k++) {
        mux_cz(controls1[k], controls2[k]);
    }
}

void CircuitBuilder::cz(QubitId v1, QubitId v2) {
    mut.op_types.push_back(OpType::CZ);
    mut.qq0.push_back(v1.tagged_id);
    mut.qq1.push_back(v2.tagged_id);
}

void CircuitBuilder::mux_cz(QubitOrXZBitOrXZBool c1, QubitOrXZBitOrXZBool c2) {
    if (c1.is_qubit_or_bit_or_bool() && c2.is_qubit_or_bit_or_bool()) {
        cz((QubitOrBitOrBool)c1, (QubitOrBitOrBool)c2);
    }
}
void CircuitBuilder::cz(QubitOrBitOrBool c1, QubitOrBitOrBool c2) {
    if (c1.is_qubit()) {
        if (c2.is_qubit()) {
            cz((QubitId)c1, (QubitId)c2);
        } else if (c2.is_bit()) {
            z_if((QubitId)c1, (BitId)c2);
        } else if ((bool)c2) {
            z((QubitId)c1);
        }
    } else if (c1.is_bit()) {
        if (c2.is_qubit()) {
            z_if((QubitId)c2, (BitId)c1);
        } else if (c2.is_bit()) {
            push_condition((BitId)c2);
            neg_if((BitId)c1);
            pop_condition();
        } else if ((bool)c2) {
            neg_if((BitId)c1);
        }
    } else if ((bool)c1) {
        z(c2);
    }
}

void CircuitBuilder::broadcast_cz(const stride_span_z &controls1, const stride_span_z &controls2) {
    mux_broadcast_cz((stride_span_xz)controls1, (stride_span_xz)controls2);
}

static inline constexpr uint8_t merge_zz_tag(QXZTypeTag8 c1, QXZTypeTag8 c2) {
    if (is_not_mixed(c1) && is_not_mixed(c2)) {
        return (uint8_t)c1 + (uint8_t)c2 * 5;
    } else {
        return 0xFF;
    }
}
void CircuitBuilder::mux_broadcast_cz(const stride_span_xz &controls1, const stride_span_xz &controls2) {
    if (controls1.size() != controls2.size()) {
        throw std::invalid_argument("broadcast_cz: len(controls1) != len(controls2)");
    }

    switch (merge_zz_tag(controls1.common_type, controls2.common_type)) {
        case merge_zz_tag(QXZTypeTag8::BOOL_VAL, QXZTypeTag8::BOOL_VAL):
            broadcast_cz(controls1.cast_data<bool>(), controls2.cast_data<bool>());
            break;
        case merge_zz_tag(QXZTypeTag8::BOOL_VAL, QXZTypeTag8::BIT_ID):
            broadcast_cz(controls1.cast_data<bool>(), controls2.cast_data<BitId>());
            break;
        case merge_zz_tag(QXZTypeTag8::BOOL_VAL, QXZTypeTag8::QUBIT_ID):
            broadcast_cz(controls1.cast_data<bool>(), controls2.cast_data<QubitId>());
            break;

        case merge_zz_tag(QXZTypeTag8::BIT_ID, QXZTypeTag8::BOOL_VAL):
            broadcast_cz(controls2.cast_data<bool>(), controls1.cast_data<BitId>());
            break;
        case merge_zz_tag(QXZTypeTag8::BIT_ID, QXZTypeTag8::BIT_ID):
            broadcast_cz(controls1.cast_data<BitId>(), controls2.cast_data<BitId>());
            break;
        case merge_zz_tag(QXZTypeTag8::BIT_ID, QXZTypeTag8::QUBIT_ID):
            broadcast_cz(controls1.cast_data<BitId>(), controls2.cast_data<QubitId>());
            break;

        case merge_zz_tag(QXZTypeTag8::QUBIT_ID, QXZTypeTag8::BOOL_VAL):
            broadcast_cz(controls2.cast_data<bool>(), controls1.cast_data<QubitId>());
            break;
        case merge_zz_tag(QXZTypeTag8::QUBIT_ID, QXZTypeTag8::BIT_ID):
            broadcast_cz(controls2.cast_data<BitId>(), controls1.cast_data<QubitId>());
            break;
        case merge_zz_tag(QXZTypeTag8::QUBIT_ID, QXZTypeTag8::QUBIT_ID):
            broadcast_cz(controls1.cast_data<QubitId>(), controls2.cast_data<QubitId>());
            break;
        default: {
            broadcast_cz(controls1.cast_data<QubitOrBitOrBool>(), controls2.cast_data<QubitOrBitOrBool>());
        }
    }
}
