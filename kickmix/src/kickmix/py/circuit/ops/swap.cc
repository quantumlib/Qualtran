#include "kickmix/build/circuit_builder.h"

using namespace kickmix;

void CircuitBuilder::broadcast_swap(
    const stride_span<const BitId> &targets1, const stride_span<const BitId> &targets2) {
    if (targets1.size() != targets2.size()) {
        throw std::invalid_argument("broadcast_swap: len(targets1) != len(targets2)");
    }
    size_t n = targets1.size();
    pattern.pattern_bit_invert_if(targets1, targets2);
    pattern.pattern_bit_invert_if(targets2, targets1);
    pattern.pattern_bit_invert_if(targets1, targets2);
    pattern.dump_into(mut, 0, n, false);
}

void CircuitBuilder::broadcast_swap(
    const stride_span<const XBitId> &targets1, const stride_span<const XBitId> &targets2) {
    broadcast_swap(targets1.cast_data<const BitId>(), targets2.cast_data<const BitId>());
}

void CircuitBuilder::broadcast_swap(
    const stride_span<const QubitId> &targets1, const stride_span<const QubitId> &targets2) {
    if (targets1.size() != targets2.size()) {
        throw std::invalid_argument("broadcast_swap: len(targets1) != len(targets2)");
    }
    size_t n = targets1.size();
    mut.qq0.push_back_many(targets1.cast_data<const uint32_t>());
    mut.qq1.push_back_many(targets2.cast_data<const uint32_t>());
    mut.op_types.push_back_repeat(OpType::SWAP, n);
}

void CircuitBuilder::swap(QubitId q1, QubitId q2) {
    mut.op_types.push_back(OpType::SWAP);
    mut.qq0.push_back(q1.tagged_id);
    mut.qq1.push_back(q2.tagged_id);
}

void CircuitBuilder::swap(BitId q1, BitId q2) {
    bit_invert_if(q1, q2);
    bit_invert_if(q2, q1);
    bit_invert_if(q1, q2);
}

void CircuitBuilder::swap(XBitId q1, XBitId q2) {
    swap(BitId(q1.untagged_id()), BitId(q2.untagged_id()));
}

void CircuitBuilder::mux_swap(QubitOrXZBitOrXZBool targets1, QubitOrXZBitOrXZBool targets2) {
    if (targets1.is_qubit() && targets2.is_qubit()) {
        swap((QubitId)targets1, (QubitId)targets2);
    } else if (targets1.is_bit() && targets2.is_bit()) {
        swap((BitId)targets1, (BitId)targets2);
    } else if (targets1.is_xbit() && targets2.is_xbit()) {
        swap((BitId)targets1, (BitId)targets2);
    } else {
        std::stringstream ss;
        ss << "Can't swap " << targets1 << " and " << targets2;
        throw std::invalid_argument(ss.str());
    }
}

void CircuitBuilder::mux_broadcast_swap(
    const stride_span<const QubitOrXZBitOrXZBool> &targets1, const stride_span<const QubitOrXZBitOrXZBool> &targets2) {
    if (targets1.size() != targets2.size()) {
        throw std::invalid_argument("broadcast_swap: len(targets1) != len(targets2)");
    }
    size_t n = targets1.size();
    for (size_t k = 0; k < n; k++) {
        mux_swap(targets1[k], targets2[k]);
    }
}

void CircuitBuilder::mux_broadcast_swap(const stride_span_xz &targets1, const stride_span_xz &targets2) {
    if (targets1.common_type == QXZTypeTag8::QUBIT_ID && targets2.common_type == QXZTypeTag8::QUBIT_ID) {
        broadcast_swap(targets1.cast_data<QubitId>(), targets2.cast_data<QubitId>());
    } else if (targets1.common_type == QXZTypeTag8::BIT_ID && targets2.common_type == QXZTypeTag8::BIT_ID) {
        broadcast_swap(targets1.cast_data<BitId>(), targets2.cast_data<BitId>());
    } else if (targets1.common_type == QXZTypeTag8::XBIT_ID && targets2.common_type == QXZTypeTag8::XBIT_ID) {
        broadcast_swap(targets1.cast_data<XBitId>(), targets2.cast_data<XBitId>());
    } else {
        mux_broadcast_swap(targets1.cast_data<QubitOrXZBitOrXZBool>(), targets2.cast_data<QubitOrXZBitOrXZBool>());
    }
}
