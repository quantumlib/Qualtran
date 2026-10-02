#include "gen_cmp_qq.h"

using namespace kickmix;

void kickmix::gen_flip_if_lt_qq(
    CircuitBuilder &builder,
    CircuitGenCtx ctx,
    stride_span<const QubitId> lhs,
    stride_span<const QubitId> rhs,
    QubitOrMinusState out,
    QubitOrBitOrBool or_equal,
    QubitOrTrue control) {
    auto mark = builder.raii_mark_block_entry("flip_if_lt_qq");

    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("gen_flip_if_lhs_lt_quantum_rhs: lhs.size() != rhs.size()");
    }
    size_t n = lhs.size();

    if (n == 0) {
        builder.ccx(control, or_equal, out);
        return;
    }
    if (n == 1) {
        builder.cx(rhs[0], lhs[0]);
        builder.parity_cccx({rhs[0], or_equal}, control, lhs[0], out, ctx.clean_workspace);
        builder.cx(rhs[0], lhs[0]);
        builder.ccx(control, or_equal, out);
        return;
    }

    if (ctx.clean_workspace.size() == 0 && !or_equal.is_qubit()) {
        throw std::invalid_argument(
            "Not enough clean workspace. gen_flip_if_lhs_lt_quantum_rhs needs 1 qubit of workspace, or or_equal to be "
            "a qubit.");
    }

    QubitId carry = or_equal.is_qubit() ? (QubitId)or_equal
                                        : ctx.take_clean(1, "carry qubit for gen_flip_if_lhs_lt_quantum_rhs")[0];
    stride_span<const QubitId> clean = ctx.clean_workspace;
    if (!or_equal.is_qubit()) {
        builder.cx(or_equal, carry);
    }
    size_t t = (size_t)std::max(0, std::min((int)clean.size() - 1, (int)n - 1));
    if (ctx.minimize_qubits) {
        t = 0;
    }
    QubitId handoff = t == 0 ? carry : clean[t];
    builder.broadcast_x(rhs);
    if (t == 0) {
        builder.x(carry);
    } else {
        builder.x(clean[0]);
        builder.cx(or_equal, clean[0]);
    }

    builder.for_each(0, t, [&](LoopBuilder &loop, iota k) {
        loop.cx(clean[k], rhs[k]);
        loop.cx(clean[k], lhs[k]);
        loop.reset(clean[k + 1]);
        loop.ccx(lhs[k], rhs[k], clean[k + 1]);
        loop.cx(clean[k], clean[k + 1]);
    });

    if (t < n - 1) {
        size_t k = t;
        builder.cx(rhs[k], lhs[k]);
        builder.cx(rhs[k], handoff);
        builder.ccx(handoff, lhs[k], rhs[k]);
    }
    builder.for_each(t + 1, n - 1, [&](LoopBuilder &loop, iota k) {
        loop.cx(rhs[k], lhs[k]);
        loop.cx(rhs[k], rhs[k - 1]);
        loop.ccx(rhs[k - 1], lhs[k], rhs[k]);
    });

    if (t == n - 1) {
        builder.ccx(control, clean[n - 1], out);
        builder.cx(clean[n - 1], lhs[n - 1]);
        builder.cx(clean[n - 1], rhs[n - 1]);
        builder.cccx(control, lhs[n - 1], rhs[n - 1], out);
        builder.cx(clean[n - 1], lhs[n - 1]);
        builder.cx(clean[n - 1], rhs[n - 1]);
    } else {
        builder.cx(rhs[n - 1], lhs[n - 1]);
        builder.cx(rhs[n - 1], rhs[n - 2]);
        builder.cccx(control, lhs[n - 1], rhs[n - 2], out);
        builder.ccx(control, rhs[n - 1], out);
        builder.cx(rhs[n - 1], lhs[n - 1]);
        builder.cx(rhs[n - 1], rhs[n - 2]);
    }

    builder.for_each_reversed(t + 1, n - 1, [&](LoopBuilder &loop, iota k) {
        loop.ccx(rhs[k - 1], lhs[k], rhs[k]);
        loop.cx(rhs[k], rhs[k - 1]);
        loop.cx(rhs[k], lhs[k]);
    });
    if (t < n - 1) {
        size_t k = t;
        builder.ccx(handoff, lhs[k], rhs[k]);
        builder.cx(rhs[k], handoff);
        builder.cx(rhs[k], lhs[k]);
    }

    {
        auto mx = builder.alloc_dirty_raii_bit();
        builder.for_each_reversed(0, t, [&](LoopBuilder &loop, iota k) {
            loop.hmr(clean[k + 1], mx.bit);
            loop.z_if(clean[k], mx.bit);
            loop.cz_if(lhs[k], rhs[k], mx.bit);
            loop.cx(clean[k], lhs[k]);
            loop.cx(clean[k], rhs[k]);
        });
    }

    if (t == 0) {
        builder.x(carry);
    } else {
        builder.x(clean[0]);
        builder.cx(or_equal, clean[0]);
    }
    builder.broadcast_x(rhs);
    builder.cx(control, out);
    if (!or_equal.is_qubit()) {
        builder.cx(or_equal, carry);
    }
}
