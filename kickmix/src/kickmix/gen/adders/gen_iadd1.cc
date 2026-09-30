#include "gen_iadd1.h"

#include "gen_iadd.h"

using namespace kickmix;

static void gen_iadd1_control_helper(
    CircuitBuilder &builder, CircuitGenCtx ctx, stride_span<const QubitId> Q_target, QubitId control) {
    size_t n = Q_target.size();

    if (n == 0) {
        return;
    }
    if (n == 1) {
        builder.cx(control, Q_target[0]);
        return;
    }
    if (n == 2) {
        builder.ccx(control, Q_target[0], Q_target[1]);
        builder.cx(control, Q_target[0]);
        return;
    }

    stride_span<const QubitId> Q_clean = ctx.take_clean(n - 2);

    builder.reset(Q_clean[0]);
    builder.ccx(control, Q_target[0], Q_clean[0]);

    builder.for_each(0, n - 3, [&](LoopBuilder &loop, iota k) {
        loop.reset(Q_clean[k + 1]);
        loop.ccx(Q_clean[k], Q_target[k + 1], Q_clean[k + 1]);
    });

    builder.ccx(Q_clean[n - 3], Q_target[n - 2], Q_target[n - 1]);

    {
        auto mx = builder.alloc_dirty_raii_bit();
        builder.for_each_reversed(0, n - 3, [&](LoopBuilder &loop, iota k) {
            loop.cx(Q_clean[k + 1], Q_target[k + 2]);
            loop.hmr(Q_clean[k + 1], mx.bit);
            loop.cz_if(Q_clean[k], Q_target[k + 1], mx.bit);
        });
    }
    builder.cx(Q_clean[0], Q_target[1]);
    builder.del_and(control, Q_target[0], Q_clean[0]);
    builder.cx(control, Q_target[0]);
}

void kickmix::gen_iadd1(
    CircuitBuilder &builder, CircuitGenCtx ctx, stride_span<const QubitId> Q_target, QubitOrTrue control) {
    if (Q_target.empty()) {
        return;
    }
    if (control.is_true()) {
        auto mark = builder.raii_mark_block_entry("iadd1");
        gen_iadd1_control_helper(builder, ctx, Q_target.subspan(1), Q_target[0]);
        builder.x(Q_target[0]);
    } else {
        auto mark = builder.raii_mark_block_entry("ciadd1");
        gen_iadd1_control_helper(builder, ctx, Q_target, (QubitId)control);
    }
}
