#include "kickmix/gen/gf_arithmetic/gen_gf_iadd.h"

#include "kickmix/build/circuit_gen_ctx.h"

using namespace kickmix;

void kickmix::gen_gf_iadd(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    stride_span<const QubitId> Q_target,
    const stride_span_z &offset,
    QubitOrTrue control) {
    (void)ctx;
    throw_unless(Q_target.size() == offset.size(), "gen_gf_iadd: Q_target.size() != Q_offset.size()");
    if (Q_target.empty()) {
        return;
    }

    if (control.is_qubit()) {
        auto mark = builder.raii_mark_block_entry("c_gf_iadd");
        builder.broadcast_ccx((QubitId)control, offset, Q_target);
    } else {
        auto mark = builder.raii_mark_block_entry("gf_iadd");
        builder.broadcast_cx(offset, (stride_span_x)Q_target);
    }
}
