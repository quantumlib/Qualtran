#include "gen_cmp.h"

#include <iostream>
#include <span>

#include "gen_cmp_low_space.h"
#include "gen_cmp_qq.h"

using namespace kickmix;

void kickmix::gen_flip_if_lt(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    stride_span<const QubitId> Q_lhs,
    const stride_span_z &rhs,
    QubitOrMinusState Q_out,
    QubitOrBitOrBool or_equal,
    QubitOrTrue control,
    double btol) {
    auto mark = builder.raii_mark_block_entry("flip_if_lt");

    size_t n = Q_lhs.size();
    throw_unless(rhs.size() == Q_lhs.size(), "rhs.size() != Q_lhs.size()");
    if (rhs.common_type == QXZTypeTag8::QUBIT_ID && btol == INFINITY && ctx.clean_workspace.size() < Q_lhs.size()) {
        gen_flip_if_lt_qq(builder, ctx, Q_lhs, rhs.cast_data<QubitId>(), Q_out, or_equal, control);
        return;
    }

    if (btol < n) {
        double n_kept = ceil(btol);
        if (n_kept <= 1) {
            // Random guessing is sufficient to get it right half of the time.
            return;
        }
        size_t offset = n - (size_t)n_kept + 1;
        gen_flip_if_lt(builder, ctx, Q_lhs.subspan(offset), rhs.skip(offset), Q_out, or_equal, control, INFINITY);
        return;
    }
    if (n == 0) {
        builder.ccx(control, or_equal, Q_out);
        return;
    } else if (n == 1) {
        builder.cx(rhs[0], Q_lhs[0]);
        builder.cccx(control, Q_lhs[0], rhs[0], Q_out, ctx.clean_workspace_if_not_minimizing());
        builder.x(Q_lhs[0]);
        builder.cccx(or_equal, control, Q_lhs[0], Q_out, ctx.clean_workspace_if_not_minimizing());
        builder.x(Q_lhs[0]);
        builder.cx(rhs[0], Q_lhs[0]);
        return;
    }
    if (ctx.minimize_qubits || ctx.clean_workspace.size() + 1 < n) {
        if (rhs.is_all_classical()) {
            gen_xif_less_than_3anc_2ntof(builder, ctx, Q_lhs, rhs, Q_out, or_equal, false, control);
            return;
        }
    }

    stride_span<const QubitId> Q_clean = ctx.clean_workspace.subspan(0, n - 1);
    auto extra_clean = ctx.clean_workspace.subspan(n - 1);
    builder.broadcast_x(Q_lhs);

    builder.broadcast_cx(rhs, Q_lhs);

    builder.reset(Q_clean[0]);
    builder.parity_ccx({or_equal, rhs[0]}, Q_lhs[0], Q_clean[0]);
    builder.cx(rhs[0], Q_clean[0]);
    builder.cx(rhs[1], Q_clean[0]);

    if (rhs.common_type == QXZTypeTag8::QUBIT_ID) {
        auto rhs_q = rhs.cast_data<QubitId>();
        builder.for_each(1, n - 1, [&](LoopBuilder &loop, iota k) {
            loop.reset(Q_clean[k]);
            loop.ccx(Q_lhs[k], Q_clean[k - 1], Q_clean[k]);
            loop.cx(rhs_q[k], Q_clean[k]);
            loop.cx(rhs_q[k + 1], Q_clean[k]);
        });
    } else if (rhs.common_type == QXZTypeTag8::BIT_ID) {
        auto rhs_b = rhs.cast_data<BitId>();
        builder.for_each(1, n - 1, [&](LoopBuilder &loop, iota k) {
            loop.reset(Q_clean[k]);
            loop.ccx(Q_lhs[k], Q_clean[k - 1], Q_clean[k]);
            loop.x_if(Q_clean[k], rhs_b[k]);
            loop.x_if(Q_clean[k], rhs_b[k + 1]);
        });
    } else {
        for (size_t k = 1; k < n - 1; k++) {
            builder.reset(Q_clean[k]);
            builder.ccx(Q_lhs[k], Q_clean[k - 1], Q_clean[k]);
            builder.cx(rhs[k], Q_clean[k]);
            builder.cx(rhs[k + 1], Q_clean[k]);
        }
    }

    builder.broadcast_cx(rhs.skip(1), Q_clean);
    {
        size_t k = n - 1;
        builder.ccx(control, rhs[k], Q_out);
        builder.cx(rhs[k], Q_clean[k - 1]);
        builder.cccx(
            control, Q_lhs[k], Q_clean[k - 1], Q_out, ctx.minimize_qubits ? std::span<const QubitId>{} : extra_clean);
        builder.cx(rhs[k], Q_clean[k - 1]);
    }

    if (rhs.common_type == QXZTypeTag8::QUBIT_ID) {
        auto rhs_q = rhs.cast_data<QubitId>();
        auto mx = builder.alloc_dirty_raii_bit();
        builder.for_each_reversed(1, n - 1, [&](LoopBuilder &loop, iota k) {
            loop.hmr(Q_clean[k], mx.bit);
            loop.push_condition(mx.bit);
            loop.z(rhs_q[k]);
            loop.cz(Q_lhs[k], Q_clean[k - 1]);
            loop.cz(Q_lhs[k], rhs_q[k]);
            loop.pop_condition();
        });
    } else if (rhs.common_type == QXZTypeTag8::BIT_ID) {
        auto rhs_b = rhs.cast_data<BitId>();
        auto mx = builder.alloc_dirty_raii_bit();
        builder.for_each_reversed(1, n - 1, [&](LoopBuilder &loop, iota k) {
            loop.hmr(Q_clean[k], mx.bit);
            loop.push_condition(mx.bit);
            loop.neg_if(rhs_b[k]);
            loop.cz(Q_lhs[k], Q_clean[k - 1]);
            loop.cz(Q_lhs[k], rhs_b[k]);
            loop.pop_condition();
        });
    } else {
        for (size_t k = n - 1; k-- > 1;) {
            auto mx = builder.hmr_raii_xbit(Q_clean[k]);
            builder.push_condition(mx.bit);
            builder.z(rhs[k]);
            builder.cz(Q_lhs[k], Q_clean[k - 1]);
            builder.cz(Q_lhs[k], rhs[k]);
            builder.pop_condition();
        }
    }
    {
        auto mx = builder.hmr_raii_xbit(Q_clean[0]);
        builder.push_condition(mx.bit);
        builder.z(rhs[0]);
        builder.cz(Q_lhs[0], or_equal);
        builder.cz(Q_lhs[0], rhs[0]);
        builder.pop_condition();
    }

    builder.broadcast_cx(rhs, Q_lhs);
    builder.broadcast_x(Q_lhs);
}

void kickmix::gen_flip_if_eq(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    stride_span<const QubitId> Q_lhs,
    const stride_span_z &rhs,
    QubitOrMinusState Q_out,
    QubitOrTrue control) {
    size_t n = Q_lhs.size();
    throw_unless(rhs.size() == Q_lhs.size(), "rhs.size() != Q_lhs.size()");
    if (n == 0) {
        builder.cx(control, Q_out);
        return;
    } else if (n == 1) {
        builder.cx(rhs[0], Q_lhs[0]);
        builder.x(Q_lhs[0]);
        builder.ccx(control, Q_lhs[0], Q_out);
        builder.x(Q_lhs[0]);
        builder.cx(rhs[0], Q_lhs[0]);
        return;
    }
    throw_unless(ctx.clean_workspace.size() >= n, "gen_flip_if_eq: ctx.clean_workspace.size() < Q_lhs.size()");

    for (size_t k = 0; k < n; k++) {
        builder.cx(rhs[k], Q_lhs[k]);
    }
    builder.broadcast_x(Q_lhs);
    auto Q_clean = ctx.clean_workspace.subspan(0, n);
    builder.reset(Q_clean[0]);
    builder.ccx(control, Q_lhs[0], Q_clean[0]);
    for (size_t k = 1; k < n; k++) {
        builder.reset(Q_clean[k]);
        builder.ccx(Q_clean[k - 1], Q_lhs[k], Q_clean[k]);
    }
    builder.ccx(Q_clean[n - 1], Q_lhs[n - 1], Q_out);
    for (size_t k = n; k-- > 1;) {
        builder.ccx(Q_clean[k - 1], Q_lhs[k], builder.hmr_raii_xbit(Q_clean[k]));
    }
    builder.ccx(control, Q_lhs[0], builder.hmr_raii_xbit(Q_clean[0]));
    builder.broadcast_x(Q_lhs);
    for (size_t k = 0; k < n; k++) {
        builder.cx(rhs[k], Q_lhs[k]);
    }
}

void kickmix::gen_flip_if_gt(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    stride_span<const QubitId> Q_lhs,
    const stride_span_z &rhs,
    QubitOrMinusState Q_out,
    QubitOrBitOrBool or_equal,
    QubitOrTrue control,
    double btol) {
    // DIDNTDO: correctness in cases where or_equal is aliased with other arguments.
    auto mark = builder.raii_mark_block_entry("flip_if_gt");

    builder.inplace_invert(or_equal);

    gen_flip_if_lt(builder, ctx, Q_lhs, rhs, Q_out, or_equal, control, btol);
    builder.cx(control, Q_out);

    builder.inplace_invert(or_equal);
}

void kickmix::gen_flip_if_ge(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    stride_span<const QubitId> Q_lhs,
    const stride_span_z &rhs,
    QubitOrMinusState Q_out,
    QubitOrTrue control,
    double btol) {
    gen_flip_if_gt(builder, ctx, Q_lhs, rhs, Q_out, true, control, btol);
}

void kickmix::gen_flip_if_le(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    stride_span<const QubitId> Q_lhs,
    const stride_span_z &rhs,
    QubitOrMinusState Q_out,
    QubitOrTrue control,
    double btol) {
    gen_flip_if_lt(builder, ctx, Q_lhs, rhs, Q_out, true, control, btol);
}
