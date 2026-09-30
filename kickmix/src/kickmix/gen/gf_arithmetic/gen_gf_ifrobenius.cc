#include "kickmix/gen/gf_arithmetic/gen_gf_ifrobenius.h"

#include "kickmix/build/circuit_gen_ctx.h"
#include "kickmix/gen/gf_arithmetic/gen_linear_map.h"

using namespace kickmix;

void kickmix::gen_gf_ifrobenius(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const QubitId> Q_target,
    size_t k) {
    size_t m = field.degree();
    throw_unless(Q_target.size() == m, "gen_gf_ifrobenius: Q_target.size() != field.degree()");
    if (m == 0 || k % m == 0) {
        return;
    }

    auto mark = builder.raii_mark_block_entry("gf_ifrobenius");
    gen_linear_map(builder, ctx, Q_target, field.frobenius_matrix(k));
}

void kickmix::gen_gf_ifrobenius_adjoint(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const QubitId> Q_target,
    size_t k) {
    size_t m = field.degree();
    throw_unless(Q_target.size() == m, "gen_gf_ifrobenius_adjoint: Q_target.size() != field.degree()");
    if (m == 0 || k % m == 0) {
        return;
    }

    auto mark = builder.raii_mark_block_entry("gf_ifrobenius_adjoint");
    // The Frobenius map has order m, so its inverse is the (m - k)-th power of itself.
    gen_linear_map(builder, ctx, Q_target, field.frobenius_matrix(m - (k % m)));
}
