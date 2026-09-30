#include "kickmix/circuit/circuit_util.h"

using namespace kickmix;

size_t kickmix::compute_reaction_depth(size_t num_qubits, size_t num_bits, std::span<const Op> operations) {
    // The reaction-depth of determining if a bit flip is present on a qubit.
    std::vector<size_t> x_depth(num_qubits);
    // The reaction-depth of determining if a phase flip is present on a qubit.
    std::vector<size_t> z_depth(num_qubits);
    // The reaction-depth of determining if a bit flip is present on a classical bit.
    std::vector<size_t> c_depth(num_bits);

    auto x_bump = [&](QubitId q, size_t d) {
        x_depth[q.untagged_id()] = std::max(x_depth[q.untagged_id()], d);
    };
    auto z_bump = [&](QubitId q, size_t d) {
        z_depth[q.untagged_id()] = std::max(z_depth[q.untagged_id()], d);
    };
    auto c_bump = [&](BitId b, size_t d) {
        c_depth[b.untagged_id()] = std::max(c_depth[b.untagged_id()], d);
    };

    std::vector<size_t> condition_depth_stack{0};
    for (const auto &op : operations) {
        size_t cond_depth = condition_depth_stack.back();
        if (op.c_condition.is_bit()) {
            cond_depth = std::max(cond_depth, c_depth[op.c_condition.untagged_id()]);
        }
        switch (op.kind) {
            case OpType::NEG:
            case OpType::NEG_IF:
            case OpType::DEBUG_PRINT_EMPTY:
            case OpType::DEBUG_PRINT_EMPTY_IF:
            case OpType::DEBUG_PRINT_C:
            case OpType::DEBUG_PRINT_C_IF:
            case OpType::DEBUG_PRINT_Q:
            case OpType::DEBUG_PRINT_Q_IF:
                // No effect.
                break;

            case OpType::X:
            case OpType::X_IF:
                x_bump(op.q_target.qubit(), cond_depth);
                break;
            case OpType::Z:
            case OpType::Z_IF:
                z_bump(op.q_target.qubit(), cond_depth);
                break;
            case OpType::BIT_INVERT:
            case OpType::BIT_INVERT_IF:
                c_bump(op.c_target.bit(), cond_depth);
                break;
            case OpType::BIT_STORE0:
            case OpType::BIT_STORE0_IF:
            case OpType::BIT_STORE1:
            case OpType::BIT_STORE1_IF:
                if (condition_depth_stack.size() == 1 && !op.c_condition.is_bit()) {
                    c_depth[op.c_target.bit().untagged_id()] = 0;
                } else {
                    c_bump(op.c_target.bit(), cond_depth);
                }
                break;
            case OpType::CX:
            case OpType::CX_IF:
                x_bump(op.q_target.qubit(), cond_depth);
                x_bump(op.q_target.qubit(), x_depth[op.q_control1.untagged_id()]);
                z_bump(op.q_control1.qubit(), cond_depth);
                z_bump(op.q_control1.qubit(), z_depth[op.q_target.untagged_id()]);
                break;
            case OpType::CZ:
            case OpType::CZ_IF:
                z_bump(op.q_target.qubit(), cond_depth);
                z_bump(op.q_target.qubit(), x_depth[op.q_control1.untagged_id()]);
                z_bump(op.q_control1.qubit(), cond_depth);
                z_bump(op.q_control1.qubit(), x_depth[op.q_target.untagged_id()]);
                break;
            case OpType::SWAP:
            case OpType::SWAP_IF:
                if (condition_depth_stack.size() == 1 && !op.c_condition.is_bit()) {
                    std::swap(x_depth[op.q_target.untagged_id()], x_depth[op.q_control1.untagged_id()]);
                    std::swap(z_depth[op.q_target.untagged_id()], z_depth[op.q_control1.untagged_id()]);
                } else {
                    x_bump(op.q_target.qubit(), x_depth[op.q_control1.untagged_id()]);
                    z_bump(op.q_target.qubit(), z_depth[op.q_control1.untagged_id()]);
                    x_bump(op.q_control1.qubit(), x_depth[op.q_target.untagged_id()]);
                    z_bump(op.q_control1.qubit(), z_depth[op.q_target.untagged_id()]);
                    z_bump(op.q_target.qubit(), cond_depth);
                    x_bump(op.q_target.qubit(), cond_depth);
                    z_bump(op.q_control1.qubit(), cond_depth);
                    x_bump(op.q_control1.qubit(), cond_depth);
                }
                break;
            case OpType::R:
            case OpType::R_IF:
                if (condition_depth_stack.size() == 1 && !op.c_condition.is_bit()) {
                    x_depth[op.q_target.untagged_id()] = 0;
                    z_depth[op.q_target.untagged_id()] = 0;
                } else {
                    x_bump(op.q_target.qubit(), cond_depth);
                }
                break;
            case OpType::HMR:
            case OpType::HMR_IF:
                if (condition_depth_stack.size() == 1 && !op.c_condition.is_bit()) {
                    c_depth[op.c_target.untagged_id()] = z_depth[op.q_target.untagged_id()] + 1;
                    x_depth[op.q_target.untagged_id()] = 0;
                    z_depth[op.q_target.untagged_id()] = 0;
                } else {
                    c_bump(op.c_target.bit(), cond_depth);
                    c_bump(op.c_target.bit(), z_depth[op.q_target.untagged_id()] + 1);
                    x_bump(op.q_target.qubit(), cond_depth);
                    z_bump(op.q_target.qubit(), cond_depth);
                }
                break;
            case OpType::CCX:
            case OpType::CCX_IF:
                x_bump(op.q_target.qubit(), cond_depth + 1);
                x_bump(op.q_target.qubit(), x_depth[op.q_control1.untagged_id()] + 1);
                x_bump(op.q_target.qubit(), x_depth[op.q_control2.untagged_id()] + 1);

                z_bump(op.q_control1.qubit(), cond_depth + 1);
                z_bump(op.q_control1.qubit(), z_depth[op.q_target.untagged_id()] + 1);
                z_bump(op.q_control1.qubit(), x_depth[op.q_control2.untagged_id()] + 1);

                z_bump(op.q_control2.qubit(), cond_depth + 1);
                z_bump(op.q_control2.qubit(), z_depth[op.q_target.untagged_id()] + 1);
                z_bump(op.q_control2.qubit(), x_depth[op.q_control1.untagged_id()] + 1);
                break;
            case OpType::Z_POW:
            case OpType::Z_POW_IF:
                if (op.angle.is_multiple_of_180_degrees()) {
                    z_bump(op.q_target.qubit(), cond_depth);
                } else {
                    z_bump(op.q_target.qubit(), cond_depth + 1);
                    z_bump(op.q_target.qubit(), x_depth[op.q_target.untagged_id()] + 1);
                }
                break;
            case OpType::CCZ:
            case OpType::CCZ_IF:
                z_bump(op.q_target.qubit(), cond_depth + 1);
                z_bump(op.q_target.qubit(), x_depth[op.q_control1.untagged_id()] + 1);
                z_bump(op.q_target.qubit(), x_depth[op.q_control2.untagged_id()] + 1);

                z_bump(op.q_control1.qubit(), cond_depth + 1);
                z_bump(op.q_control1.qubit(), x_depth[op.q_target.untagged_id()] + 1);
                z_bump(op.q_control1.qubit(), x_depth[op.q_control2.untagged_id()] + 1);

                z_bump(op.q_control2.qubit(), cond_depth + 1);
                z_bump(op.q_control2.qubit(), x_depth[op.q_target.untagged_id()] + 1);
                z_bump(op.q_control2.qubit(), x_depth[op.q_control1.untagged_id()] + 1);
                break;
            case OpType::PUSH_CONDITION:
                condition_depth_stack.push_back(cond_depth);
                break;
            case OpType::POP_CONDITION:
                condition_depth_stack.pop_back();
                break;
            default:
                throw std::invalid_argument("operation type not handled in reaction_depth: " + op.str());
        }
    }
    size_t max_depth = 0;
    for (const auto &e : x_depth) {
        max_depth = std::max(max_depth, e);
    }
    for (const auto &e : z_depth) {
        max_depth = std::max(max_depth, e);
    }
    for (const auto &e : c_depth) {
        max_depth = std::max(max_depth, e);
    }
    return max_depth;
}

size_t kickmix::compute_num_touched_qubits(const Circuit &circuit) {
    std::vector<bool> used(circuit.num_qubits);
    for (size_t k = 0; k < circuit.num_q_ops; k++) {
        used[circuit.q0[k]] = true;
    }
    for (size_t k = 0; k < circuit.num_qq_ops; k++) {
        used[circuit.qq0[k]] = true;
        used[circuit.qq1[k]] = true;
    }
    for (size_t k = 0; k < circuit.num_qqq_ops; k++) {
        used[circuit.qqq0[k]] = true;
        used[circuit.qqq1[k]] = true;
        used[circuit.qqq2[k]] = true;
    }
    size_t total = 0;
    for (auto e : used) {
        total += e;
    }
    return total;
}
