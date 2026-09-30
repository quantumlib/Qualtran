#include "kickmix/gen/gf_arithmetic/gen_gf_div.h"

#include "kickmix/build/circuit_gen_ctx.h"
#include "kickmix/gen/gf_arithmetic/gen_gf_inverse.h"
#include "kickmix/gen/gf_arithmetic/gen_gf_mul.h"

using namespace kickmix;

size_t kickmix::gf_div_workspace_size(const GF2Field &field) {
    return field.degree() + gf_inverse_chain_size(field);
}

void kickmix::gen_gf_div(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const QubitId> Q_target,
    stride_span<const QubitId> Q_lhs,
    stride_span<const QubitId> Q_rhs) {
    size_t m = field.degree();
    throw_unless(Q_target.size() == m, "gen_gf_div: Q_target.size() != field.degree()");
    throw_unless(Q_lhs.size() == m, "gen_gf_div: Q_lhs.size() != field.degree()");
    throw_unless(Q_rhs.size() == m, "gen_gf_div: Q_rhs.size() != field.degree()");

    auto mark = builder.raii_mark_block_entry("gf_div");
    CircuitGenCtx sub_ctx = ctx;
    auto Q_inverse = sub_ctx.take_clean(m, "gen_gf_div_inv");
    auto Q_chain = sub_ctx.take_clean(gf_inverse_chain_size(field), "gen_gf_div_chain");
    builder.broadcast_reset(Q_inverse);
    builder.broadcast_reset(Q_chain);

    gen_gf_inverse(builder, sub_ctx, field, Q_inverse, Q_rhs, Q_chain);
    gen_gf_mul(builder, sub_ctx.with_more_dirty_qubits(Q_chain), field, Q_target, Q_lhs, Q_inverse);
    gen_gf_uninverse(builder, sub_ctx, field, Q_inverse, Q_rhs, Q_chain);
}

void kickmix::gen_gf_div(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const QubitId> Q_target,
    stride_span<const QubitId> Q_lhs,
    stride_span<const QubitId> Q_rhs,
    stride_span<const QubitId> Q_ancillas) {
    size_t m = field.degree();
    size_t chain_size = gf_inverse_chain_size(field);
    throw_unless(Q_target.size() == m, "gen_gf_div: Q_target.size() != field.degree()");
    throw_unless(Q_lhs.size() == m, "gen_gf_div: Q_lhs.size() != field.degree()");
    throw_unless(Q_rhs.size() == m, "gen_gf_div: Q_rhs.size() != field.degree()");
    throw_unless(
        Q_ancillas.size() == m + chain_size,
        "gen_gf_div: Q_ancillas.size() != field.degree() + gf_inverse_chain_size(field)");

    auto mark = builder.raii_mark_block_entry("gf_div");
    auto Q_inverse = Q_ancillas.keep(m);
    auto Q_chain = Q_ancillas.skip(m);
    builder.broadcast_reset(Q_inverse);
    if (chain_size > 0) {
        builder.broadcast_reset(Q_chain);
    }

    gen_gf_inverse(builder, ctx, field, Q_inverse, Q_rhs, Q_chain);
    gen_gf_mul(builder, ctx.with_more_dirty_qubits(Q_chain), field, Q_target, Q_lhs, Q_inverse);
}

void kickmix::gen_gf_undiv(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const QubitId> Q_target,
    stride_span<const QubitId> Q_lhs,
    stride_span<const QubitId> Q_rhs) {
    size_t m = field.degree();
    throw_unless(Q_target.size() == m, "gen_gf_undiv: Q_target.size() != field.degree()");
    throw_unless(Q_lhs.size() == m, "gen_gf_undiv: Q_lhs.size() != field.degree()");
    throw_unless(Q_rhs.size() == m, "gen_gf_undiv: Q_rhs.size() != field.degree()");

    auto mark = builder.raii_mark_block_entry("gf_undiv");
    CircuitGenCtx sub_ctx = ctx;
    auto Q_inverse = sub_ctx.take_clean(m, "gen_gf_undiv_inv");
    auto Q_chain = sub_ctx.take_clean(gf_inverse_chain_size(field), "gen_gf_undiv_chain");
    builder.broadcast_reset(Q_inverse);
    builder.broadcast_reset(Q_chain);

    // Recomputing the divisor's inverse is unavoidable, but clearing the quotient on top of it is
    // free, because gen_gf_unmul pays for it with measurements instead of Toffolis.
    gen_gf_inverse(builder, sub_ctx, field, Q_inverse, Q_rhs, Q_chain);
    gen_gf_unmul(builder, sub_ctx.with_more_dirty_qubits(Q_chain), field, Q_target, Q_lhs, Q_inverse);
    gen_gf_uninverse(builder, sub_ctx, field, Q_inverse, Q_rhs, Q_chain);
}

void kickmix::gen_gf_undiv(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const QubitId> Q_target,
    stride_span<const QubitId> Q_lhs,
    stride_span<const QubitId> Q_rhs,
    stride_span<const QubitId> Q_ancillas) {
    size_t m = field.degree();
    size_t chain_size = gf_inverse_chain_size(field);
    throw_unless(Q_target.size() == m, "gen_gf_undiv: Q_target.size() != field.degree()");
    throw_unless(Q_lhs.size() == m, "gen_gf_undiv: Q_lhs.size() != field.degree()");
    throw_unless(Q_rhs.size() == m, "gen_gf_undiv: Q_rhs.size() != field.degree()");
    throw_unless(
        Q_ancillas.size() == m + chain_size,
        "gen_gf_undiv: Q_ancillas.size() != field.degree() + gf_inverse_chain_size(field)");

    auto mark = builder.raii_mark_block_entry("gf_undiv");
    auto Q_inverse = Q_ancillas.keep(m);
    auto Q_chain = Q_ancillas.skip(m);

    gen_gf_unmul(builder, ctx.with_more_dirty_qubits(Q_chain), field, Q_target, Q_lhs, Q_inverse);
    gen_gf_uninverse(builder, ctx, field, Q_inverse, Q_rhs, Q_chain);
}
