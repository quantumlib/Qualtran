#include "kickmix/gen/gf_arithmetic/gen_gf_imul_classical.h"

#include "kickmix/build/circuit_gen_ctx.h"
#include "kickmix/gen/gf_arithmetic/gen_linear_map.h"

using namespace kickmix;

void kickmix::gen_gf_imul_classical(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const QubitId> Q_target,
    const GF2Poly &constant) {
    size_t m = field.degree();
    throw_unless(Q_target.size() == m, "gen_gf_imul_classical: Q_target.size() != field.degree()");
    GF2Poly c = field.mod(constant);
    throw_unless(!c.is_zero(), "gen_gf_imul_classical: multiplication by zero is not reversible");
    if (c == field.one()) {
        return;
    }

    auto mark = builder.raii_mark_block_entry("gf_imul_classical");
    gen_linear_map(builder, ctx, Q_target, field.constant_mul_matrix(c));
}

void kickmix::gen_gf_idiv_classical(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const QubitId> Q_target,
    const GF2Poly &constant) {
    size_t m = field.degree();
    throw_unless(Q_target.size() == m, "gen_gf_idiv_classical: Q_target.size() != field.degree()");
    GF2Poly c = field.mod(constant);
    throw_unless(!c.is_zero(), "gen_gf_idiv_classical: division by zero is not reversible");
    if (c == field.one()) {
        return;
    }

    auto mark = builder.raii_mark_block_entry("gf_idiv_classical");
    // Inverting the constant classically is cheaper than inverting the synthesized matrix.
    gen_linear_map(builder, ctx, Q_target, field.constant_mul_matrix(field.invert(c)));
}
