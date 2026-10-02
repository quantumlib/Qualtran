#include "gen_ineg_mod.h"

#include "kickmix/gen/adders/gen_iadd.h"
#include "kickmix/gen/adders/gen_iadd_classical.h"
#include "kickmix/gen/comparators/gen_cmp.h"

using namespace kickmix;

void kickmix::gen_ineg_mod(
    CircuitBuilder &builder, CircuitGenCtx ctx, std::span<const QubitId> target, const stride_span_z &modulus) {
    size_t n = target.size();
    throw_unless(modulus.size() > 0, "modulus.size() == 0 (division by zero will occur because modulus must be zero)");
    throw_unless(modulus.size() == n, "modulus.size() != target.size()");
    throw_unless(modulus.back() == true, "modulus must be full capacity (modulus[target.size() - 1] == true)");

    auto mark = builder.raii_mark_block_entry("ineg_mod");

    for (auto q : target) {
        builder.x(q);
    }
    gen_iadd_classical(builder, ctx, target, modulus, true, true, INFINITY);

    // The input 0 has become equal to the modulus. Conditionally xor it back into 0.
    auto ctx_original = ctx;
    auto q = ctx.take_clean(1);
    builder.broadcast_reset(q);
    gen_flip_if_eq(builder, ctx, target, modulus, q[0], true);
    for (size_t k = 0; k < modulus.size(); k++) {
        builder.ccx(q[0], modulus[k], target[k]);
    }
    {
        auto mx = builder.hmr_raii_push_condition(q[0]);
        gen_flip_if_eq(builder, ctx_original, target, stride_span_z::repeat_false(n), MINUS_KET, true);
    }
}
