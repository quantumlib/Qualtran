#include "gen_iadd_control.h"

#include <span>

using namespace kickmix;

void kickmix::gen_iadd_control(
    CircuitBuilder &builder,
    CircuitGenCtx ctx,
    stride_span<const QubitId> Q_target,
    stride_span<const QubitId> Q_offset,
    QubitOrBitOrBool carry_in,
    QubitId control) {
    auto mark = builder.raii_mark_block_entry("iadd_control");

    size_t n = Q_target.size();
    if (n > Q_offset.size()) {
        throw std::invalid_argument("Q_target.size() > Q_offset.size()");
    }
    if (n == 0) {
        return;
    }
    if (n == 1) {
        builder.ccx(control, Q_offset[0], Q_target[0]);
        builder.ccx(control, carry_in, Q_target[0]);
        return;
    }
    stride_span<const QubitId> Q_clean = ctx.take_clean(n, "gen_iadd_control");

    {
        builder.cx(carry_in, Q_target[0]);
        builder.cx(carry_in, Q_offset[0]);
        builder.reset(Q_clean[0]);
        builder.ccx(Q_target[0], Q_offset[0], Q_clean[0]);
        builder.cx(carry_in, Q_clean[0]);

        builder.for_each(1, n - 1, [&](LoopBuilder &loop, iota k) {
            loop.cx(Q_clean[k - 1], Q_target[k]);
            loop.cx(Q_clean[k - 1], Q_offset[k]);
            loop.reset(Q_clean[k]);
            loop.ccx(Q_target[k], Q_offset[k], Q_clean[k]);
            loop.cx(Q_clean[k - 1], Q_clean[k]);
        });
    }

    builder.cx(Q_offset[n - 1], Q_clean[n - 2]);
    builder.ccx(control, Q_clean[n - 2], Q_target[n - 1]);
    builder.cx(Q_offset[n - 1], Q_clean[n - 2]);

    {
        auto mx = builder.alloc_dirty_raii_bit();
        builder.for_each_reversed(1, n - 1, [&](LoopBuilder &loop, iota k) {
            loop.hmr(Q_clean[k], mx.bit);
            loop.cz_if(Q_target[k], Q_offset[k], mx.bit);
            loop.z_if(Q_clean[k - 1], mx.bit);
            loop.ccx(control, Q_offset[k], Q_target[k]);
            loop.cx(Q_clean[k - 1], Q_target[k]);
            loop.cx(Q_clean[k - 1], Q_offset[k]);
        });
    }

    {
        auto mx = builder.hmr_raii_xbit(Q_clean[0]);
        builder.ccx(Q_target[0], Q_offset[0], mx);
        builder.cx(carry_in, mx);
        builder.ccx(control, Q_offset[0], Q_target[0]);
        builder.cx(carry_in, Q_target[0]);
        builder.cx(carry_in, Q_offset[0]);
    }
}

void kickmix::gen_isub_control(
    CircuitBuilder &builder,
    CircuitGenCtx ctx,
    stride_span<const QubitId> Q_target,
    stride_span<const QubitId> Q_offset,
    QubitId control) {
    auto mark = builder.raii_mark_block_entry("isub_control");
    builder.broadcast_x(Q_target);
    gen_iadd_control(builder, ctx, Q_target, Q_offset, false, control);
    builder.broadcast_x(Q_target);
}