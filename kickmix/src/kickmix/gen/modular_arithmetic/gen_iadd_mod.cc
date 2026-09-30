#include "gen_iadd_mod.h"

#include <span>

#include "kickmix/gen/adders/gen_iadd.h"
#include "kickmix/gen/adders/gen_iadd_classical.h"
#include "kickmix/gen/comparators/gen_cmp.h"

using namespace kickmix;

void kickmix::gen_iadd_mod(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    std::span<const QubitId> target,
    std::span<const QubitId> offset,
    const stride_span_z &modulus,
    QubitOrTrue control,
    double btol) {
    size_t n = modulus.size();
    throw_unless(target.size() == n, "gen_iadd_mod: target.size() != modulus.size()");
    throw_unless(offset.size() == n, "gen_iadd_mod: offset.size() != modulus.size()");
    throw_unless(ctx.clean_workspace.size() >= 2, "gen_iadd_mod: need more clean qubits");

    auto mark = builder.raii_mark_block_entry("iadd_mod");
    btol += 1;

    std::vector<QubitId> extended_target;
    for (auto q : target) {
        extended_target.push_back(q);
    }
    extended_target.push_back(ctx.clean_workspace[0]);

    array_z extended_modulus = array_z::copy_of_concat(modulus, stride_span_z::repeat_false(1));

    gen_iadd(builder, ctx.with_clean_subspan(1), extended_target, offset, control);
    auto cmp = ctx.clean_workspace[1];
    builder.reset(cmp);
    {
        auto sub_ctx = ctx.with_clean_subspan(2).with_more_dirty_qubits(offset);
        gen_flip_if_ge(builder, sub_ctx, extended_target, extended_modulus, cmp, true, btol);
        gen_isub_classical(builder, sub_ctx, extended_target, extended_modulus, false, cmp, btol);
    }
    builder.del_zero(ctx.clean_workspace[0]);
    {
        auto mx = builder.hmr_raii_push_condition(cmp);
        gen_flip_if_lt(builder, ctx, target, offset, MINUS_KET, false, control, btol);
    }
}

void kickmix::gen_isub_mod(
    CircuitBuilder &builder,
    CircuitGenCtx ctx,
    std::span<const QubitId> target,
    std::span<const QubitId> offset,
    const stride_span_z &modulus,
    QubitOrTrue control,
    double btol) {
    size_t n = modulus.size();
    throw_unless(target.size() == n, "target.size() != modulus.size()");
    throw_unless(offset.size() == n, "offset.size() != modulus.size()");

    auto mark = builder.raii_mark_block_entry("isub_mod");

    std::vector<QubitId> extended_target;
    for (auto q : target) {
        extended_target.push_back(q);
    }
    auto target_sign = ctx.take_clean(1)[0];
    extended_target.push_back(target_sign);

    array_z extended_modulus = array_z::copy_of_concat(modulus, stride_span_z::repeat_false(1));

    std::vector<QubitId> extended_offset;
    for (auto q : offset) {
        extended_offset.push_back(q);
    }
    auto offset_sign = ctx.take_clean(1)[0];
    extended_offset.push_back(offset_sign);

    builder.reset(extended_target.back());
    btol += 1;

    // Unconditionally subtract, with an extra qubit to detect underflow.
    gen_isub(builder, ctx, extended_target, offset, control);

    // Conditioned on an underflow having occurred, normalize by adding the modulus.
    {
        auto sub_ctx = ctx.with_more_dirty_qubits(offset);
        gen_iadd_classical(builder, sub_ctx, target, modulus, false, extended_target.back(), btol);
    }

    // Measurement-based-uncomputation of the underflow qubit.
    {
        auto mark2 = builder.raii_mark_block_entry("mbuc underflow");
        {
            auto mx = builder.hmr_raii_xbit(extended_target.back());
            builder.push_condition(mx.bit);
        }
        builder.reset(extended_target.back());
        gen_iadd(builder, ctx, extended_target, extended_offset);
        gen_flip_if_ge(builder, ctx, extended_target, extended_modulus, MINUS_KET, control, btol);
        gen_isub(builder, ctx, extended_target, extended_offset);
        builder.pop_condition();
    }
    builder.del_zero(target_sign);
    builder.del_zero(offset_sign);
}
