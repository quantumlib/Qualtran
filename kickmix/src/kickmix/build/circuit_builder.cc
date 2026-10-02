#include "kickmix/build/circuit_builder.h"

#include <iostream>

using namespace kickmix;

std::vector<QubitId> CircuitBuilder::append_register(size_t length, std::string_view name) {
    std::vector<QubitId> result;
    for (uint32_t k = 0; k < length; k++) {
        result.push_back(QubitId{next_qubit_id});
        next_qubit_id += 1;
    }

    mut.register_data.push_back({});
    mut.register_data.back().name = name;
    append_qubit_to_register(result, RegisterId{next_register_id});
    next_register_id += 1;
    return result;
}

std::vector<QubitOrBitOrBool> CircuitBuilder::append_register_mixed_result(size_t length, std::string_view name) {
    auto res = append_register(length, name);
    std::vector<QubitOrBitOrBool> result;
    for (auto e : res) {
        result.push_back(e);
    }
    return result;
}

std::vector<QubitOrBitOrBool> CircuitBuilder::append_classical_register_mixed_result(
    size_t length, std::string_view name) {
    auto res = append_classical_register(length, name);
    std::vector<QubitOrBitOrBool> result;
    for (auto e : res) {
        result.push_back(e);
    }
    return result;
}

array_z CircuitBuilder::append_classical_register_qcarray_result(size_t length, std::string_view name) {
    return array_z::copy_of(append_classical_register(length, name));
}

array_z CircuitBuilder::append_register_qcarray_result(size_t length, std::string_view name) {
    return array_z::copy_of(append_register(length, name));
}

std::vector<BitId> CircuitBuilder::reserve_bits(size_t length) {
    std::vector<BitId> result;
    for (uint32_t k = 0; k < length; k++) {
        result.push_back(BitId{next_bit_id});
        next_bit_id += 1;
    }
    return result;
}

RegisterId CircuitBuilder::append_classical_register(std::span<const BitId> bits, std::string_view name) {
    mut.register_data.push_back({});
    mut.register_data.back().name = name;

    RegisterId result{next_register_id};
    append_bit_to_register(bits, result);
    next_register_id += 1;
    return result;
}

std::vector<BitId> CircuitBuilder::append_classical_register(size_t length, std::string_view name) {
    mut.register_data.push_back({});
    mut.register_data.back().name = name;

    std::vector<BitId> result = reserve_bits(length);
    append_bit_to_register(result, RegisterId{next_register_id});
    next_register_id += 1;
    return result;
}

RegisterId CircuitBuilder::reserve_register() {
    RegisterId r = RegisterId{next_register_id};
    next_register_id += 1;
    return r;
}

std::vector<QubitId> CircuitBuilder::reserve_qubits(size_t length) {
    std::vector<QubitId> result;
    for (uint32_t k = 0; k < length; k++) {
        result.push_back(QubitId{next_qubit_id});
        next_qubit_id += 1;
    }
    return result;
}

CircuitBuilderRaiiBit CircuitBuilder::alloc_dirty_raii_bit() {
    BitId bit;
    if (raii_bits.empty()) {
        bit = BitId{next_bit_id};
        next_bit_id++;
    } else {
        bit = raii_bits.back();
        raii_bits.pop_back();
    }
    return CircuitBuilderRaiiBit(this, bit, false);
}

CircuitBuilderRaiiBit CircuitBuilder::alloc_clean_raii_bit() {
    BitId bit;
    if (raii_bits.empty()) {
        bit = BitId{next_bit_id};
        next_bit_id++;
    } else {
        bit = raii_bits.back();
        raii_bits.pop_back();
        bit_store0(bit);
    }
    return CircuitBuilderRaiiBit(this, bit, false);
}

void CircuitBuilder::append_bit_to_register(std::span<const BitId> bits, RegisterId reg) {
    while (mut.register_data.size() <= reg.id) {
        mut.register_data.push_back({});
    }
    for (auto b : bits) {
        mut.register_data[reg.id].contents.push_back(b);
    }
}
void CircuitBuilder::append_qubit_to_register(std::span<const QubitId> qubits, RegisterId reg) {
    while (mut.register_data.size() <= reg.id) {
        mut.register_data.push_back({});
    }
    for (auto q : qubits) {
        mut.register_data[reg.id].contents.push_back(q);
    }
}
void CircuitBuilder::x(QubitOrXBitOrXBool target) {
    if (target.is_qubit()) {
        x((QubitId)target);
    } else if (target.is_xbit()) {
        neg_if(((XBitId)target).conjugated_by_h());
    } else {
        z(target.is_minus_ket());
    }
}
void CircuitBuilder::x(QubitId target) {
    mut.op_types.push_back(OpType::X);
    mut.q0.push_back(target.tagged_id);
}

void CircuitBuilder::x_if(QubitId target, BitId cond) {
    mut.op_types.push_back(OpType::X_IF);
    mut.bc.push_back(cond.tagged_id);
    mut.q0.push_back(target.tagged_id);
}

void CircuitBuilder::broadcast_x(stride_span<const QubitId> targets) {
    mut.op_types.push_back_repeat(OpType::X, targets.size());
    mut.q0.push_back_many(targets.cast_data<const uint32_t>());
}

CircuitBuilderRaiiXBit CircuitBuilder::hmr_raii_xbit(QubitId target) {
    CircuitBuilderRaiiXBit out = CircuitBuilderRaiiXBit(std::move(alloc_dirty_raii_bit()));
    hmr(target, out.bit);
    return out;
}

CircuitBuilderRaiiXBit::CircuitBuilderRaiiXBit(CircuitBuilder *builder, BitId bit, bool push)
    : builder(builder), bit(bit), is_pushed(push) {
    if (push && builder != nullptr) {
        builder->push_condition(bit);
    }
}
CircuitBuilderRaiiXBit::~CircuitBuilderRaiiXBit() {
    if (builder != nullptr) {
        if (is_pushed) {
            builder->pop_condition();
            is_pushed = false;
        }
        builder->raii_bits.push_back(bit);
    }
    builder = nullptr;
}
CircuitBuilderRaiiBit::CircuitBuilderRaiiBit(CircuitBuilder *builder, BitId bit, bool push)
    : builder(builder), bit(bit), is_pushed(push) {
    if (push && builder != nullptr) {
        builder->push_condition(bit);
    }
}
CircuitBuilderRaiiBit::~CircuitBuilderRaiiBit() {
    if (builder != nullptr) {
        if (is_pushed) {
            builder->pop_condition();
            is_pushed = false;
        }
        builder->raii_bits.push_back(bit);
    }
    builder = nullptr;
}

CircuitBuilderRaiiXBit CircuitBuilder::hmr_raii_push_condition(QubitId target) {
    CircuitBuilderRaiiXBit out = CircuitBuilderRaiiXBit(std::move(alloc_dirty_raii_bit()));
    hmr(target, out.bit);
    push_condition(out.bit);
    out.is_pushed = true;
    return out;
}

void CircuitBuilder::hmr(QubitId target, BitId out) {
    mut.op_types.push_back(OpType::HMR);
    mut.q0.push_back(target.tagged_id);
    mut.b0.push_back(out.tagged_id);
}
void CircuitBuilder::hmr_if(QubitId target, BitId out, BitId cond) {
    mut.op_types.push_back(OpType::HMR_IF);
    mut.bc.push_back(cond.tagged_id);
    mut.q0.push_back(target.tagged_id);
    mut.b0.push_back(out.tagged_id);
}

void CircuitBuilder::reset(QubitId target) {
    mut.op_types.push_back(OpType::R);
    mut.q0.push_back(target.tagged_id);
}
void CircuitBuilder::reset_if(QubitId target, BitId cond) {
    mut.op_types.push_back(OpType::R_IF);
    mut.bc.push_back(cond.tagged_id);
    mut.q0.push_back(target.tagged_id);
}
void CircuitBuilder::broadcast_reset(stride_span<const QubitId> targets) {
    mut.op_types.push_back_repeat(OpType::R, targets.size());
    mut.q0.push_back_many(targets.cast_data<const uint32_t>());
}

void CircuitBuilder::cleft_rotate(QubitOrTrue control, stride_span<const QubitId> reg) {
    auto mark = raii_mark_block_entry(control.is_qubit() ? "c_left_rotate" : "left_rotate");
    if (reg.empty()) {
        return;
    }

    if (control.is_qubit()) {
        for_each_reversed(0, reg.size() - 1, [&](LoopBuilder &loop, iota k) {
            loop.cx(reg[k], reg[k + 1]);
            loop.ccx((QubitId)control, reg[k + 1], reg[k]);
            loop.cx(reg[k], reg[k + 1]);
        });
    } else {
        for_each_reversed(0, reg.size() - 1, [&](LoopBuilder &loop, iota k) {
            loop.swap(reg[k], reg[k + 1]);
        });
    }
}

RaiiCircuitBuilderPopPushedCondition::RaiiCircuitBuilderPopPushedCondition(CircuitBuilder *builder) : builder(builder) {
}
RaiiCircuitBuilderPopPushedCondition::~RaiiCircuitBuilderPopPushedCondition() {
    if (builder != nullptr) {
        builder->pop_condition();
        builder = nullptr;
    }
}
RaiiCircuitBuilderPopPushedCondition::RaiiCircuitBuilderPopPushedCondition(
    RaiiCircuitBuilderPopPushedCondition &&other) noexcept
    : builder(other.builder) {
    other.builder = nullptr;
}
RaiiCircuitBuilderPopPushedCondition &RaiiCircuitBuilderPopPushedCondition::operator=(
    RaiiCircuitBuilderPopPushedCondition &&other) noexcept {
    builder = other.builder;
    other.builder = nullptr;
    return *this;
}

void CircuitBuilder::inplace_invert(QubitOrBitOrBool &e) {
    if (e.is_bool()) {
        e = !(bool)e;
    } else if (e.is_qubit()) {
        x((QubitId)e);
    } else {
        bit_invert((BitId)e);
    }
}
void CircuitBuilder::inplace_invert(stride_span<QubitOrBitOrBool> target) {
    for (auto &e : target) {
        inplace_invert(e);
    }
}
void CircuitBuilder::inplace_invert(array_z &target) {
    for (size_t k = 0; k < target.count; ++k) {
        inplace_invert(target[k]);
    }
}
void CircuitBuilder::broadcast_ccz(
    QubitOrBitOrBool control1, const stride_span_z &controls2, stride_span<const QubitId> targets) {
    size_t n = std::min(targets.size(), controls2.size());
    if (control1.is_qubit()) {
        if (controls2.common_type == QXZTypeTag8::BOOL_VAL) {
            auto *out = mut.qq1.grab_writeable(n);
            uint32_t *cur_out = out;
            for (size_t k = 0; k < n; k++) {
                *cur_out = targets[k].tagged_id;
                cur_out += (bool)controls2[k];
            }
            mut.qq1.rewind_tail(cur_out);
            size_t count = cur_out - out;
            mut.qq0.push_back_repeat(control1.tagged_id, count);
            mut.op_types.push_back_repeat(OpType::CZ, count);
        } else if (controls2.common_type == QXZTypeTag8::BIT_ID) {
            for_each(0, n, [&](LoopBuilder &loop, iota k) {
                loop.cz_if((QubitId)control1, targets[k], controls2.cast_data<BitId>()[k]);
            });
        } else if (controls2.common_type == QXZTypeTag8::QUBIT_ID) {
            for_each(0, n, [&](LoopBuilder &loop, iota k) {
                loop.ccz((QubitId)control1, controls2.cast_data<QubitId>()[k], targets[k]);
            });
        } else {
            for (size_t k = 0; k < n; k++) {
                auto c = controls2[k];
                if (c.is_qubit()) {
                    ccz((QubitId)control1, (QubitId)c, targets[k]);
                } else if (c.is_bit()) {
                    cz_if((QubitId)control1, targets[k], (BitId)c);
                } else if (c == true) {
                    cz((QubitId)control1, targets[k]);
                }
            }
        }
    } else if (control1.is_bit()) {
        auto raii_cond = raii_push_condition((BitId)control1);
        broadcast_cz(controls2, targets);
    } else if (control1 == true) {
        broadcast_cz(controls2, targets);
    }
}
void CircuitBuilder::broadcast_ccx(
    QubitOrBitOrBool control1, const stride_span_z &controls2, stride_span<const QubitId> targets) {
    size_t n = std::min(targets.size(), controls2.size());
    if (control1.is_qubit()) {
        if (controls2.common_type == QXZTypeTag8::BOOL_VAL) {
            auto *out = mut.qq1.grab_writeable(n);
            auto *cur_out = out;
            for (size_t k = 0; k < n; k++) {
                *cur_out = targets[k].tagged_id;
                cur_out += (bool)controls2[k];
            }
            mut.qq1.rewind_tail(cur_out);
            size_t count = cur_out - out;
            mut.qq0.push_back_repeat(control1.tagged_id, count);
            mut.op_types.push_back_repeat(OpType::CX, count);
        } else if (controls2.common_type == QXZTypeTag8::BIT_ID) {
            for_each(0, n, [&](LoopBuilder &loop, iota k) {
                loop.cx_if((QubitId)control1, targets[k], controls2.cast_data<BitId>()[k]);
            });
        } else if (controls2.common_type == QXZTypeTag8::QUBIT_ID) {
            for_each(0, n, [&](LoopBuilder &loop, iota k) {
                loop.ccx((QubitId)control1, controls2.cast_data<QubitId>()[k], targets[k]);
            });
        } else {
            for (size_t k = 0; k < n; k++) {
                auto c = controls2[k];
                if (c.is_qubit()) {
                    ccx((QubitId)control1, (QubitId)c, targets[k]);
                } else if (c.is_bit()) {
                    cx_if((QubitId)control1, targets[k], (BitId)c);
                } else if (c == true) {
                    cx((QubitId)control1, targets[k]);
                }
            }
        }
    } else if (control1.is_bit()) {
        auto raii_cond = raii_push_condition((BitId)control1);
        mux_broadcast_cx((stride_span_xz)controls2, (stride_span_xz)targets);
    } else if (control1 == true) {
        broadcast_cx((stride_span_z)controls2, (stride_span_x)targets);
    }
}

void CircuitBuilder::cright_rotate(QubitOrTrue control, stride_span<const QubitId> reg) {
    auto mark = raii_mark_block_entry(control.is_qubit() ? "c_right_rotate" : "right_rotate");
    if (reg.empty()) {
        return;
    }

    if (control.is_qubit()) {
        for_each(0, reg.size() - 1, [&](LoopBuilder &loop, iota k) {
            loop.cx(reg[k], reg[k + 1]);
            loop.ccx((QubitId)control, reg[k + 1], reg[k]);
            loop.cx(reg[k], reg[k + 1]);
        });
    } else {
        for_each(0, reg.size() - 1, [&](LoopBuilder &loop, iota k) {
            loop.swap(reg[k], reg[k + 1]);
        });
    }
}
RaiiCircuitBuilderPopPushedCondition CircuitBuilder::raii_push_condition(BitId control) {
    push_condition(control);
    return RaiiCircuitBuilderPopPushedCondition(this);
}
void CircuitBuilder::left_rotate(stride_span<const QubitId> reg) {
    for_each_reversed(0, reg.size() - 1, [&](LoopBuilder &loop, iota k) {
        loop.swap(reg[k], reg[k + 1]);
    });
}
void CircuitBuilder::right_rotate(stride_span<const QubitId> reg) {
    for_each(0, reg.size() - 1, [&](LoopBuilder &loop, iota k) {
        loop.swap(reg[k], reg[k + 1]);
    });
}

void CircuitBuilder::cswap(BitId control, QubitId q1, QubitId q2) {
    mut.op_types.push_back(OpType::SWAP_IF);
    mut.bc.push_back(control.tagged_id);
    mut.qq0.push_back(q1.tagged_id);
    mut.qq1.push_back(q2.tagged_id);
}

void CircuitBuilder::cswap(QubitOrBitOrBool control, QubitId q1, QubitId q2) {
    if (control.is_qubit()) {
        cx(q1, q2);
        ccx((QubitId)control, q2, q1);
        cx(q1, q2);
    } else if (control.is_bit()) {
        cswap((BitId)control, q1, q2);
    } else if ((bool)control) {
        swap(q1, q2);
    }
}

void CircuitBuilder::broadcast_hmr(stride_span<const QubitId> target, stride_span<const BitId> output) {
    if (target.size() != output.size()) {
        throw std::invalid_argument("target.size() != output.size()");
    }
    mut.op_types.push_back_repeat(OpType::HMR, target.size());
    mut.q0.push_back_many(target.cast_data<const uint32_t>());
    mut.b0.push_back_many(output.cast_data<const uint32_t>());
}

void CircuitBuilder::broadcast_cswap(
    QubitOrTrue control, stride_span<const QubitId> q1, stride_span<const QubitId> q2) {
    if (q1.size() != q2.size()) {
        throw std::invalid_argument("cswap between registers of different sizes.");
    }

    if (control.is_qubit()) {
        for_each(0, q1.size(), [&](LoopBuilder &loop, iota k) {
            loop.cx(q1[k], q2[k]);
            loop.ccx((QubitId)control, q2[k], q1[k]);
            loop.cx(q1[k], q2[k]);
        });
    } else {
        for_each(0, q1.size(), [&](LoopBuilder &loop, iota k) {
            loop.swap(q1[k], q2[k]);
        });
    }
}

void CircuitBuilder::cx_if(QubitId control, QubitId target, BitId cond) {
    mut.op_types.push_back(OpType::CX_IF);
    mut.bc.push_back(cond.tagged_id);
    mut.qq0.push_back(control.tagged_id);
    mut.qq1.push_back(target.tagged_id);
}
void CircuitBuilder::ccx(QubitOrBitOrBool control2, QubitOrBitOrBool control1, QubitOrXBitOrXBool target) {
    if (control2.is_qubit()) {
        if (control1.is_qubit()) {
            if (target.is_qubit()) {
                ccx((QubitId)control2, (QubitId)control1, (QubitId)target);
            } else if (target.is_xbit()) {
                cz_if((QubitId)control2, (QubitId)control1, ((XBitId)target).conjugated_by_h());
            } else if (target.is_minus_ket()) {
                cz((QubitId)control2, (QubitId)control1);
            }
        } else if (control1.is_bit()) {
            if (target.is_qubit()) {
                cx_if((QubitId)control2, (QubitId)target, (BitId)control1);
            } else if (target.is_xbit()) {
                push_condition(((XBitId)target).conjugated_by_h());
                z_if((QubitId)control2, (BitId)control1);
                pop_condition();
            } else if (target.is_minus_ket()) {
                z_if((QubitId)control2, (BitId)control1);
            }
        } else if ((bool)control1) {
            if (target.is_qubit()) {
                cx((QubitId)control2, (QubitId)target);
            } else if (target.is_xbit()) {
                z_if((QubitId)control2, ((XBitId)target).conjugated_by_h());
            } else if (target.is_minus_ket()) {
                z((QubitId)control2);
            }
        }
    } else if (control2.is_bit()) {
        if (control1.is_qubit()) {
            if (target.is_qubit()) {
                cx_if((QubitId)control1, (QubitId)target, (BitId)control2);
            } else if (target.is_xbit()) {
                push_condition(((XBitId)target).conjugated_by_h());
                z_if((QubitId)control1, (BitId)control2);
                pop_condition();
            } else if (target.is_minus_ket()) {
                z_if((QubitId)control1, (BitId)control2);
            }
        } else if (control1.is_bit()) {
            push_condition((BitId)control2);
            if (target.is_qubit()) {
                x_if((QubitId)target, (BitId)control1);
            } else if (target.is_xbit()) {
                push_condition(((XBitId)target).conjugated_by_h());
                neg_if((BitId)control1);
                pop_condition();
            } else if (target.is_minus_ket()) {
                neg_if((BitId)control1);
            }
            pop_condition();
        } else if ((bool)control1) {
            if (target.is_qubit()) {
                x_if((QubitId)target, (BitId)control2);
            } else if (target.is_xbit()) {
                push_condition(((XBitId)target).conjugated_by_h());
                neg_if((BitId)control2);
                pop_condition();
            } else if (target.is_minus_ket()) {
                neg_if((BitId)control2);
            }
        }
    } else if ((bool)control2) {
        cx(control1, target);
    }
}
void CircuitBuilder::ccz(QubitOrBitOrBool c1, QubitOrBitOrBool c2, QubitOrBitOrBool c3) {
    if (c1.is_qubit()) {
        if (c2.is_qubit()) {
            if (c3.is_qubit()) {
                ccz((QubitId)c1, (QubitId)c2, (QubitId)c3);
            } else if (c3.is_bit()) {
                cz_if((QubitId)c1, (QubitId)c2, (BitId)c3);
            } else if ((bool)c3) {
                cz((QubitId)c1, (QubitId)c2);
            }
        } else if (c2.is_bit()) {
            if (c3.is_qubit()) {
                cz_if((QubitId)c1, (QubitId)c3, (BitId)c2);
            } else if (c3.is_bit()) {
                push_condition((BitId)c3);
                z_if((QubitId)c1, (BitId)c2);
                pop_condition();
            } else if ((bool)c3) {
                z_if((QubitId)c1, (BitId)c2);
            }
        } else if ((bool)c2) {
            if (c3.is_qubit()) {
                cz((QubitId)c1, (QubitId)c3);
            } else if (c3.is_bit()) {
                z_if((QubitId)c1, (BitId)c3);
            } else if ((bool)c3) {
                z((QubitId)c1);
            }
        }
    } else if (c1.is_bit()) {
        if (c2.is_qubit()) {
            if (c3.is_qubit()) {
                cz_if((QubitId)c2, (QubitId)c3, (BitId)c1);
            } else if (c3.is_bit()) {
                push_condition((BitId)c1);
                z_if((QubitId)c2, (BitId)c3);
                pop_condition();
            } else if ((bool)c3) {
                z_if((QubitId)c2, (BitId)c1);
            }
        } else if (c2.is_bit()) {
            push_condition((BitId)c1);
            if (c3.is_qubit()) {
                z_if((QubitId)c3, (BitId)c2);
            } else if (c3.is_bit()) {
                push_condition((BitId)c2);
                neg_if((BitId)c3);
                pop_condition();
            } else if ((bool)c3) {
                neg_if((BitId)c2);
            }
            pop_condition();
        } else if ((bool)c2) {
            if (c3.is_qubit()) {
                z_if((QubitId)c3, (BitId)c1);
            } else if (c3.is_bit()) {
                push_condition((BitId)c1);
                neg_if((BitId)c3);
                pop_condition();
            } else if ((bool)c3) {
                neg_if((BitId)c1);
            }
        }
    } else if ((bool)c1) {
        cz(c2, c3);
    }
}

void CircuitBuilder::push_condition(BitId bit) {
    mut.op_types.push_back(OpType::PUSH_CONDITION);
    mut.bc.push_back(bit.tagged_id);
}

void CircuitBuilder::pop_condition() {
    mut.op_types.push_back(OpType::POP_CONDITION);
}

void CircuitBuilder::cccx(
    QubitId control3, QubitId control2, QubitId control1, QubitId target, std::span<const QubitId> clean) {
    if (!clean.empty()) {
        reset(clean[0]);
        ccx(control3, control2, clean[0]);
        ccx(clean[0], control1, target);
        cz_if(control3, control2, hmr_raii_xbit(clean[0]).bit);
        return;
    }

    // Find a dirty helper qubit to use (use smallest possible index to avoid increasing qubit count).
    QubitId helper;
    while (control3 == helper || control2 == helper || control1 == helper || target == helper) {
        helper.tagged_id++;
    }
    ccx(control3, control2, helper);
    ccx(helper, control1, target);
    ccx(control3, control2, helper);
    ccx(helper, control1, target);
}

void CircuitBuilder::cccz(
    QubitId control3, QubitId control2, QubitId control1, QubitId control0, std::span<const QubitId> clean) {
    if (!clean.empty()) {
        reset(clean[0]);
        ccx(control3, control2, clean[0]);
        ccz(clean[0], control1, control0);
        cz_if(control3, control2, hmr_raii_xbit(clean[0]).bit);
        return;
    }

    // Find a dirty helper qubit to use (use smallest possible index to avoid increasing qubit count).
    QubitId helper;
    while (control3 == helper || control2 == helper || control1 == helper || control0 == helper) {
        helper.tagged_id++;
    }
    ccx(control3, control2, helper);
    ccz(helper, control1, control0);
    ccx(control3, control2, helper);
    ccz(helper, control1, control0);
}

void CircuitBuilder::c_push(BitOrBool control) {
    if (control.is_bit()) {
        c_push((BitId)control);
    } else {
        c_push((bool)control);
    }
}

void CircuitBuilder::c_push(QubitId control) {
    c_resolve_qubit_controls.push_back(control);
}

void CircuitBuilder::c_push(BitId control) {
    c_resolve_bit_controls.push_back(control);
}

void CircuitBuilder::c_push(QubitOrTrue control) {
    if (control.is_qubit()) {
        c_resolve_qubit_controls.push_back((QubitId)control);
    }
}

void CircuitBuilder::c_push(bool control) {
    c_resolve_has_false_control |= !control;
}

void CircuitBuilder::c_push(QubitOrBitOrBool control) {
    if (control.is_qubit()) {
        c_resolve_qubit_controls.push_back((QubitId)control);
    } else if (control.is_bit()) {
        c_resolve_bit_controls.push_back((BitId)control);
    } else if (control == false) {
        c_resolve_has_false_control = true;
    }
}

std::invalid_argument CircuitBuilder::c_resolve_explain_failure(std::string_view msg) const {
    std::stringstream ss;
    ss << msg << "\n";
    ss << "\nRequested Controls {";
    for (const auto &e : c_resolve_qubit_controls) {
        ss << "\n    " << e;
    }
    for (const auto &e : c_resolve_bit_controls) {
        ss << "\n    " << e;
    }
    if (c_resolve_has_false_control) {
        ss << "\n    false";
    }
    ss << "\n}";
    return std::invalid_argument(ss.str());
}

void CircuitBuilder::c_resolve(const CircuitBuilderRaiiXBit &target, std::span<const QubitId> clean) {
    c_resolve_bit_controls.push_back(target.bit);
    c_resolve_exact(MINUS_KET, clean);
}
void CircuitBuilder::c_resolve(QubitId target, std::span<const QubitId> clean) {
    c_resolve_exact(target, clean);
}
void CircuitBuilder::c_resolve_exact(QubitOrMinusState target, std::span<const QubitId> clean) {
    if (!c_resolve_has_false_control) {
        for (size_t k = 1; k < c_resolve_bit_controls.size(); k++) {
            push_condition(c_resolve_bit_controls[k]);
        }

        const auto &qs = c_resolve_qubit_controls;

        if (target.is_qubit()) {
            if (c_resolve_bit_controls.empty()) {
                switch (c_resolve_qubit_controls.size()) {
                    case 3:
                        cccx(qs[0], qs[1], qs[2], (QubitId)target, clean);
                        break;
                    case 2:
                        ccx(qs[0], qs[1], (QubitId)target);
                        break;
                    case 1:
                        cx(qs[0], (QubitId)target);
                        break;
                    case 0:
                        x((QubitId)target);
                        break;
                    default:
                        throw c_resolve_explain_failure("Not implemented: cc..cx had more than 3 controls.");
                }
            } else {
                BitId cond = c_resolve_bit_controls[0];
                switch (c_resolve_qubit_controls.size()) {
                    case 3:
                        push_condition(cond);
                        cccx(qs[0], qs[1], qs[2], (QubitId)target, clean);
                        pop_condition();
                        break;
                    case 2:
                        ccx_if(qs[0], qs[1], (QubitId)target, cond);
                        break;
                    case 1:
                        cx_if(qs[0], (QubitId)target, cond);
                        break;
                    case 0:
                        x_if((QubitId)target, cond);
                        break;
                    default:
                        throw c_resolve_explain_failure("Not implemented: cc..cx had more than 3 controls.");
                }
            }
        } else {
            if (c_resolve_bit_controls.empty()) {
                switch (c_resolve_qubit_controls.size()) {
                    case 4:
                        cccz(qs[0], qs[1], qs[2], qs[3], clean);
                        break;
                    case 3:
                        ccz(qs[0], qs[1], qs[2]);
                        break;
                    case 2:
                        cz(qs[0], qs[1]);
                        break;
                    case 1:
                        z(qs[0]);
                        break;
                    case 0:
                        neg();
                        break;
                    default:
                        throw c_resolve_explain_failure("Not implemented: cc..cz had more than 4 controls.");
                }
            } else {
                BitId cond = c_resolve_bit_controls[0];
                switch (c_resolve_qubit_controls.size()) {
                    case 4:
                        push_condition(cond);
                        cccz(qs[0], qs[1], qs[2], qs[3], clean);
                        pop_condition();
                        break;
                    case 3:
                        ccz_if(qs[0], qs[1], qs[2], cond);
                        break;
                    case 2:
                        cz_if(qs[0], qs[1], cond);
                        break;
                    case 1:
                        z_if(qs[0], cond);
                        break;
                    case 0:
                        neg_if(cond);
                        break;
                    default:
                        throw c_resolve_explain_failure("Not implemented: cc..cz had more than 4 controls.");
                }
            }
        }
        for (size_t k = 1; k < c_resolve_bit_controls.size(); k++) {
            pop_condition();
        }
    }

    c_resolve_bit_controls.clear();
    c_resolve_qubit_controls.clear();
    c_resolve_has_false_control = false;
}
void CircuitBuilder::c_resolve(QubitOrXBitOrXBool target, std::span<const QubitId> clean) {
    if (target.is_xbit()) {
        c_resolve_bit_controls.push_back(((XBitId)target).conjugated_by_h());
        c_resolve_exact(MINUS_KET, clean);
    } else if (target.is_qubit()) {
        c_resolve_exact((QubitId)target, clean);
    } else if (target.is_plus_ket()) {
        c_resolve_has_false_control = true;
    } else if (target.is_minus_ket()) {
        c_resolve_exact(MINUS_KET, clean);
    }
}

void CircuitBuilder::parity_cccx(
    const std::array<QubitOrBitOrBool, 2> &c_parity,
    QubitOrBitOrBool c2,
    QubitOrBitOrBool c3,
    QubitOrXBitOrXBool target,
    std::span<const QubitId> clean) {
    for (size_t k = 0; k < 2; k++) {
        if (c_parity[k].is_qubit()) {
            if (c_parity[k] == c2 || c_parity[k] == c3) {
                throw std::invalid_argument("c_parity[k] == c2 || c_parity[k] == c3");
            }
            cx(c_parity[k ^ 1], (QubitId)c_parity[k]);
            cccx((QubitId)c_parity[k], c2, c3, target, clean);
            cx(c_parity[k ^ 1], (QubitId)c_parity[k]);
            return;
        }
    }

    // Both parity controls are classical, so this is at worst just two CCX operations.
    cccx(c_parity[0], c2, c3, target, clean);
    cccx(c_parity[1], c2, c3, target, clean);
}
void CircuitBuilder::parity_cccz(
    const std::array<QubitOrBitOrBool, 2> &c_parity,
    QubitOrBitOrBool c2,
    QubitOrBitOrBool c3,
    QubitOrBitOrBool target,
    std::span<const QubitId> clean) {
    for (size_t k = 0; k < 2; k++) {
        if (c_parity[k].is_qubit()) {
            if (c_parity[k] == c2 || c_parity[k] == c3) {
                throw std::invalid_argument("c_parity[k] == c2 || c_parity[k] == c3");
            }
            cx(c_parity[k ^ 1], (QubitId)c_parity[k]);
            cccz((QubitId)c_parity[k], c2, c3, target, clean);
            cx(c_parity[k ^ 1], (QubitId)c_parity[k]);
            return;
        }
    }

    // Both parity controls are classical, so this is at worst just two CCZ operations.
    cccz(c_parity[0], c2, c3, target, clean);
    cccz(c_parity[1], c2, c3, target, clean);
}
void CircuitBuilder::parity_ccx(
    const std::array<QubitOrBitOrBool, 2> &c_parity, QubitOrBitOrBool c_single, QubitOrXBitOrXBool target) {
    if (c_parity[0] == c_parity[1]) {
        // Cancels to false.
        return;
    }

    if (c_parity[0] == c_single || c_parity[1] == c_single) {
        ccx(c_parity[0], c_parity[1], target);
        cx(c_single, target);
        return;
    }

    for (size_t k = 0; k < 2; k++) {
        if (c_parity[k].is_qubit()) {
            cx(c_parity[k ^ 1], (QubitId)c_parity[k]);
            ccx((QubitId)c_parity[k], c_single, target);
            cx(c_parity[k ^ 1], (QubitId)c_parity[k]);
            return;
        }
    }

    // Both parity controls are classical, so this is at worst just two CX operations.
    ccx(c_parity[0], c_single, target);
    ccx(c_parity[1], c_single, target);
}
void CircuitBuilder::parity_ccz(
    const std::array<QubitOrBitOrBool, 2> &c_parity, QubitOrBitOrBool c_single, QubitOrBitOrBool target) {
    if (c_parity[0] == c_parity[1]) {
        // Cancels to false.
        return;
    }

    if (c_parity[0] == c_single || c_parity[1] == c_single) {
        ccz(c_parity[0], c_parity[1], target);
        cz(c_single, target);
        return;
    }

    for (size_t k = 0; k < 2; k++) {
        if (c_parity[k].is_qubit()) {
            cx(c_parity[k ^ 1], (QubitId)c_parity[k]);
            ccz((QubitId)c_parity[k], c_single, target);
            cx(c_parity[k ^ 1], (QubitId)c_parity[k]);
            return;
        }
    }

    // Both parity controls are classical, so this is at worst just two CZ operations.
    ccz(c_parity[0], c_single, target);
    ccz(c_parity[1], c_single, target);
}
void CircuitBuilder::ccx(QubitId control2, QubitId control1, QubitId target) {
    mut.op_types.push_back(OpType::CCX);
    mut.qqq0.push_back(control2.tagged_id);
    mut.qqq1.push_back(control1.tagged_id);
    mut.qqq2.push_back(target.tagged_id);
}
void CircuitBuilder::ccx_if(QubitId control2, QubitId control1, QubitId target, BitId cond) {
    mut.op_types.push_back(OpType::CCX_IF);
    mut.bc.push_back(cond.tagged_id);
    mut.qqq0.push_back(control2.tagged_id);
    mut.qqq1.push_back(control1.tagged_id);
    mut.qqq2.push_back(target.tagged_id);
}
void CircuitBuilder::ccz(QubitId control2, QubitId control1, QubitId target) {
    mut.op_types.push_back(OpType::CCZ);
    mut.qqq0.push_back(control2.tagged_id);
    mut.qqq1.push_back(control1.tagged_id);
    mut.qqq2.push_back(target.tagged_id);
}
void CircuitBuilder::ccz_if(QubitId control2, QubitId control1, QubitId target, BitId cond) {
    mut.op_types.push_back(OpType::CCZ_IF);
    mut.bc.push_back(cond.tagged_id);
    mut.qqq0.push_back(control2.tagged_id);
    mut.qqq1.push_back(control1.tagged_id);
    mut.qqq2.push_back(target.tagged_id);
}

void CircuitBuilder::reset_and(QubitOrBitOrBool control1, QubitOrBitOrBool control2, QubitId target) {
    reset(target);
    ccx(control1, control2, target);
}

void CircuitBuilder::del_zero(QubitId target) {
    hmr_raii_xbit(target);
}
void CircuitBuilder::broadcast_del_zero(stride_span<const QubitId> target) {
    for (auto q : target) {
        hmr_raii_xbit(q);
    }
}

void CircuitBuilder::del_and(QubitOrBitOrBool control1, QubitOrBitOrBool control2, QubitId target) {
    auto mx = hmr_raii_xbit(target);
    ccx(control1, control2, mx);
}

void CircuitBuilder::cz_if(QubitId v1, QubitId v2, BitId cond) {
    mut.op_types.push_back(OpType::CZ_IF);
    mut.bc.push_back(cond.tagged_id);
    mut.qq0.push_back(v1.tagged_id);
    mut.qq1.push_back(v2.tagged_id);
}
void CircuitBuilder::z(QubitOrBitOrBool v) {
    if (v.is_qubit()) {
        z((QubitId)v);
    } else if (v.is_bit()) {
        z((BitId)v);
    } else if ((bool)v) {
        neg();
    }
}
void CircuitBuilder::z(BitOrBool v) {
    if (v.is_bit()) {
        z((BitId)v);
    } else {
        z((bool)v);
    }
}
void CircuitBuilder::z(BitId v) {
    neg_if(v);
}
void CircuitBuilder::z(bool v) {
    if (v) {
        neg();
    }
}

void CircuitBuilder::debug_print(QubitOrXZBitOrXZBool q) {
    if (q.is_qubit()) {
        debug_print((QubitId)q);
    } else if (q.is_bit()) {
        debug_print((BitId)q);
    } else if (q.is_xbit()) {
        debug_print(BitId(q.untagged_id()));
    } else if (q.is_xbool()) {
        debug_print(q.is_minus_ket());
    } else {
        debug_print((bool)q);
    }
}
void CircuitBuilder::debug_print(BitOrBool q) {
    if (q.is_bit()) {
        debug_print((BitId)q);
    } else {
        debug_print((bool)q);
    }
}
void CircuitBuilder::debug_print(bool q) {
    auto b = alloc_dirty_raii_bit();
    if (q) {
        bit_store1(b.bit);
    } else {
        bit_store0(b.bit);
    }
    debug_print(b.bit);
}
void CircuitBuilder::debug_print(BitId b) {
    mut.op_types.push_back(OpType::DEBUG_PRINT_C);
    mut.b0.push_back(b.tagged_id);
}
void CircuitBuilder::debug_print_if(QubitOrBit qb, BitId cond) {
    if (qb.is_qubit()) {
        debug_print_if((QubitId)qb, cond);
    } else {
        debug_print_if((BitId)qb, cond);
    }
}
void CircuitBuilder::debug_print_if(RegisterId r, BitId cond) {
    if (r.id < mut.register_data.size()) {
        for (const auto &e : mut.register_data[r.id].contents) {
            debug_print_if(e, cond);
        }
        debug_print_if(cond);
    }
}
void CircuitBuilder::debug_print() {
    mut.op_types.push_back(OpType::DEBUG_PRINT_EMPTY);
}
void CircuitBuilder::debug_print_if(BitId cond) {
    mut.op_types.push_back(OpType::DEBUG_PRINT_EMPTY_IF);
    mut.bc.push_back(cond.tagged_id);
}
void CircuitBuilder::debug_print(std::span<const QubitId> q) {
    for (size_t k = q.size(); k--;) {
        debug_print(q[k]);
    }
    debug_print();
}
void CircuitBuilder::debug_print(std::span<const BitOrBool> q) {
    for (size_t k = q.size(); k--;) {
        debug_print(q[k]);
    }
    debug_print();
}
void CircuitBuilder::debug_print(std::span<const QubitOrBitOrBool> q) {
    for (size_t k = q.size(); k--;) {
        debug_print(q[k]);
    }
    debug_print();
}
void CircuitBuilder::debug_print(QubitId q) {
    mut.op_types.push_back(OpType::DEBUG_PRINT_Q);
    mut.q0.push_back(q.tagged_id);
}
void CircuitBuilder::debug_print_if(BitId b, BitId cond) {
    mut.op_types.push_back(OpType::DEBUG_PRINT_C_IF);
    mut.b0.push_back(b.tagged_id);
    mut.bc.push_back(cond.tagged_id);
}
void CircuitBuilder::debug_print_if(QubitId q, BitId cond) {
    mut.op_types.push_back(OpType::DEBUG_PRINT_Q_IF);
    mut.q0.push_back(q.tagged_id);
    mut.bc.push_back(cond.tagged_id);
}
void CircuitBuilder::bit_invert(BitId v) {
    mut.op_types.push_back(OpType::BIT_INVERT);
    mut.b0.push_back(v.tagged_id);
}
void CircuitBuilder::bit_store0(BitId v) {
    mut.op_types.push_back(OpType::BIT_STORE0);
    mut.b0.push_back(v.tagged_id);
}
void CircuitBuilder::bit_store1(BitId v) {
    mut.op_types.push_back(OpType::BIT_STORE1);
    mut.b0.push_back(v.tagged_id);
}

void CircuitBuilder::bit_invert_if(BitId v, BitId condition) {
    mut.op_types.push_back(OpType::BIT_INVERT_IF);
    mut.bc.push_back(condition.tagged_id);
    mut.b0.push_back(v.tagged_id);
}
void CircuitBuilder::bit_store0_if(BitId v, BitId condition) {
    mut.op_types.push_back(OpType::BIT_STORE0_IF);
    mut.bc.push_back(condition.tagged_id);
    mut.b0.push_back(v.tagged_id);
}
void CircuitBuilder::bit_store1_if(BitId v, BitId condition) {
    mut.op_types.push_back(OpType::BIT_STORE1_IF);
    mut.bc.push_back(condition.tagged_id);
    mut.b0.push_back(v.tagged_id);
}

void CircuitBuilder::bit_store_and(BitId c1, BitId c2, BitId target) {
    bit_store0(target);
    push_condition(c1);
    bit_store1_if(target, c2);
    pop_condition();
}

void CircuitBuilder::z(QubitId v1) {
    mut.op_types.push_back(OpType::Z);
    mut.q0.push_back(v1.tagged_id);
}
void CircuitBuilder::z_if(QubitId v1, BitId cond) {
    mut.op_types.push_back(OpType::Z_IF);
    mut.bc.push_back(cond.tagged_id);
    mut.q0.push_back(v1.tagged_id);
}
void CircuitBuilder::z_pow(QubitId v1, FixedPrecisionAngle128 exponent) {
    mut.op_types.push_back(OpType::Z_POW);
    mut.q0.push_back(v1.tagged_id);
    mut.angles.push_back(exponent);
}
void CircuitBuilder::z_pow_if(QubitId v1, FixedPrecisionAngle128 exponent, BitId cond) {
    mut.op_types.push_back(OpType::Z_POW_IF);
    mut.bc.push_back(cond.tagged_id);
    mut.q0.push_back(v1.tagged_id);
    mut.angles.push_back(exponent);
}
void CircuitBuilder::neg() {
    mut.op_types.push_back(OpType::NEG);
}
void CircuitBuilder::neg_if(BitId cond) {
    mut.op_types.push_back(OpType::NEG_IF);
    mut.bc.push_back(cond.tagged_id);
}

Circuit CircuitBuilder::finish_circuit() const {
    return mut.to_validated_circuit();
}

void CircuitBuilder::write_analysis_svg_to(std::ostream &out, size_t reference_qubit_count) {
    Circuit circuit = finish_circuit();

    // Perform circuit analysis.
    std::vector<uint64_t> active_qubits((circuit.num_qubits + 63) / 64);
    std::vector<std::string_view> name_stack;
    std::vector<size_t> toffoli_stack;
    std::vector<std::vector<uint64_t>> qubit_touched_stack;
    qubit_touched_stack.push_back(std::vector<uint64_t>((circuit.num_qubits + 63) / 64));
    size_t k_op = 0;
    size_t toffoli_count = 0;
    std::vector<std::string> group_names;
    std::vector<size_t> group_heights;
    std::vector<size_t> group_toffoli_counts;
    std::vector<size_t> group_max_active_qubits;
    std::vector<size_t> group_hits;
    std::string cur_name;
    std::vector<Op> ops;
    circuit.iter_ops([&](const Op &op) {
        ops.push_back(op);
    });
    auto padded_marks = marks;
    padded_marks.insert(padded_marks.begin(), Mark{true, "<entire circuit>", 0});
    padded_marks.push_back(Mark{false, "<entire circuit>", ops.size()});
    for (const auto &mark : padded_marks) {
        while (k_op < mark.offset) {
            const auto &op = ops[k_op];

            // Track which qubits were touched.
            for (auto q : std::array<QubitOrFalse, 3>{op.q_target, op.q_control1, op.q_control2}) {
                if (q.is_qubit()) {
                    qubit_touched_stack.back()[q.untagged_id() / 64] |= uint64_t{1} << (q.untagged_id() % 64);
                }
            }

            // Track which qubits are active (have been touched since last HMR).
            switch (op.kind) {
                case OpType::HMR:
                case OpType::HMR_IF:
                    active_qubits[op.q_target.untagged_id() / 64] &= ~(uint64_t{1} << (op.q_target.untagged_id() % 64));
                    break;
                default:
                    for (auto q : std::array<QubitOrFalse, 3>{op.q_target, op.q_control1, op.q_control2}) {
                        if (q.is_qubit()) {
                            active_qubits[q.untagged_id() / 64] |= uint64_t{1} << (q.untagged_id() % 64);
                        }
                    }
            }

            // Count certain operation types.
            switch (op.kind) {
                case OpType::CCX:
                case OpType::CCZ:
                case OpType::CCX_IF:
                case OpType::CCZ_IF:
                    toffoli_count += 1;
                    break;
                case OpType::Z_POW:
                case OpType::Z_POW_IF:
                    if (!op.angle.is_multiple_of_90_degrees()) {
                        toffoli_count += 1;  // TODO: better handle the conversion from Z_POW to Toffoli
                    }
                    break;
                case OpType::NEG_IF:
                case OpType::BIT_INVERT_IF:
                case OpType::BIT_STORE0_IF:
                case OpType::BIT_STORE1_IF:
                case OpType::NEG:
                case OpType::BIT_INVERT:
                case OpType::BIT_STORE0:
                case OpType::BIT_STORE1:
                case OpType::PUSH_CONDITION:
                case OpType::POP_CONDITION:
                case OpType::DEBUG_PRINT_EMPTY_IF:
                case OpType::DEBUG_PRINT_C:
                case OpType::DEBUG_PRINT_EMPTY:
                case OpType::DEBUG_PRINT_C_IF:
                case OpType::DEBUG_PRINT_Q:
                case OpType::DEBUG_PRINT_Q_IF:
                case OpType::HMR:
                case OpType::HMR_IF:
                case OpType::R_IF:
                case OpType::X_IF:
                case OpType::Z_IF:
                case OpType::CX_IF:
                case OpType::CZ_IF:
                case OpType::SWAP_IF:
                case OpType::R:
                case OpType::X:
                case OpType::Z:
                case OpType::CX:
                case OpType::CZ:
                case OpType::SWAP:
                    break;
                default:
                    throw std::invalid_argument("Unhandled operation kind: " + op.str());
            }
            k_op++;
        }

        if (mark.enter_else_exit) {
            cur_name.append(":");
            cur_name.append(mark.name);
            name_stack.push_back(mark.name);
            toffoli_stack.push_back(toffoli_count);
            qubit_touched_stack.push_back(active_qubits);
        } else {
            if (name_stack.empty() || name_stack.back() != mark.name) {
                throw std::invalid_argument("stack error");
            }

            // Find this stack's group.
            size_t k_group = 0;
            while (k_group < group_names.size()) {
                if (group_names[k_group] == cur_name) {
                    break;
                }
                k_group++;
            }
            if (k_group == group_names.size()) {
                group_names.push_back(cur_name);
                group_toffoli_counts.push_back(0);
                group_heights.push_back(name_stack.size());
                group_max_active_qubits.push_back(0);
                group_hits.push_back(0);
            }

            // Update group stats.
            group_toffoli_counts[k_group] += toffoli_count - toffoli_stack.back();
            auto popped_touched = std::move(qubit_touched_stack.back());
            size_t max_active_qubits = 0;
            for (size_t k = 0; k < popped_touched.size(); k++) {
                max_active_qubits += std::popcount(popped_touched[k]);
            }
            group_max_active_qubits[k_group] = std::max(group_max_active_qubits[k_group], max_active_qubits);
            group_hits[k_group] += 1;

            // Pop stack frame.
            for (size_t k = 0; k < mark.name.size() + 1; k++) {
                cur_name.pop_back();
            }
            toffoli_stack.pop_back();
            name_stack.pop_back();
            qubit_touched_stack.pop_back();
        }
    }

    out << R"SVG(<svg viewBox="0 0 1000.0 200.0" xmlns="http://www.w3.org/2000/svg">)SVG" << "\n";
    double x_scale = 1000.0 / toffoli_count;
    double y_scale = 100.0 / circuit.num_qubits;
    std::vector<size_t> layer_xs{0};
    for (size_t k = 0; k < group_names.size(); k++) {
        auto h = group_heights[k];
        while (layer_xs.size() <= h) {
            layer_xs.push_back(0);
        }
        size_t layer_x = layer_xs[h];
        for (size_t k2 = 0; k2 < h; k2++) {
            layer_x = std::max(layer_xs[k2], layer_x);
        }
        layer_xs[h] = layer_x + group_toffoli_counts[k];
        std::string_view group_name = group_names[k];
        group_name = group_name.substr(group_name.rfind(':') + 1);
        out << "<rect";
        out << " x=\"" << layer_x * x_scale << "\"";
        out << " y=\"" << -y_scale * group_max_active_qubits[k] << "\"";
        out << " width=\"" << group_toffoli_counts[k] * x_scale << "\"";
        out << " height=\"" << y_scale * group_max_active_qubits[k] << "\"";
        out << " stroke=\"black\"";
        out << " fill=\"blue\"";
        out << " />\n";
        out << "<rect";
        out << " x=\"" << layer_x * x_scale << "\"";
        out << " y=\"" << h * 10 - 10 << "\"";
        out << " width=\"" << group_toffoli_counts[k] * x_scale << "\"";
        out << " height=\"" << 10 << "\"";
        out << " stroke=\"black\"";
        out << " fill=\"red\"";
        out << " opacity=\"0.8\"";
        out << " />\n";
        out << "<text";
        out << " x=\"" << (layer_x + group_toffoli_counts[k] * 0.5) * x_scale << "\"";
        out << " y=\"" << h * 10 - 3 << "\"";
        std::string title;
        title.append(group_name);
        if (group_hits[k] > 1) {
            title.append(" (x");
            title.append(std::to_string(group_hits[k]));
            title.append(")");
        }
        float box_width = group_toffoli_counts[k] * x_scale;
        float font_size = box_width / title.size() * 2;
        font_size = std::min(5.0f, font_size);
        font_size = std::max(0.5f, font_size);
        out << " font-size=\"" << font_size << "\"";
        out << " dominant-baseline=\"bottom\"";
        out << " text-anchor=\"middle\"";
        out << " fill=\"black\"";
        out << " >" << title;
        out << "</text>\n";
    }
    out << "<text";
    out << " x=\"500\"";
    out << " y=\"-100\"";
    out << " font-size=\"10\"";
    out << " dominant-baseline=\"bottom\"";
    out << " text-anchor=\"middle\"";
    out << " fill=\"black\"";
    out << " >" << circuit.num_qubits << " qubits, " << circuit.max_magic() << " toffolis</text>\n";
    if (reference_qubit_count > 0) {
        out << "<path";
        out << " d=\"";
        for (size_t k = reference_qubit_count; k < circuit.num_qubits; k += reference_qubit_count) {
            out << " M500," << -y_scale * k << " l50,0";
        }
        out << "\"";
        out << " fill=\"none\"";
        out << " stroke=\"green\"";
        out << " />\n";
    }

    out << "</svg>";
}

RaiiDeferredMark::~RaiiDeferredMark() {
    if (builder != nullptr) {
        builder->marks.push_back(deferred);
        builder->marks.back().offset = builder->mut.op_types.size();
        builder = nullptr;
    }
}

RaiiDeferredMark CircuitBuilder::raii_mark_block_entry(std::string_view name) {
    marks.push_back({
        true,
        name,
        mut.op_types.size(),
    });
    return RaiiDeferredMark(this, Mark{false, name, 0});
}

void CircuitBuilder::for_each(size_t start, size_t end, const std::function<void(LoopBuilder &loop, iota k)> &func) {
    if (start >= end) {
        return;
    }
    LoopBuilder loop{&pattern};
    func(loop, iota{0});
    pattern.dump_into(mut, start, end, false);
}

void CircuitBuilder::for_each_reversed(
    size_t start, size_t end, const std::function<void(LoopBuilder &loop, iota k)> &func) {
    if (start >= end) {
        return;
    }
    LoopBuilder loop{&pattern};
    func(loop, iota{0});
    pattern.dump_into(mut, start, end, true);
}
