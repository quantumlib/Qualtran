#include "gen_idouble_mod_approx.h"

#include <span>

#include "kickmix/gen/adders/gen_iadd_classical.h"

using namespace kickmix;

void kickmix::gen_idouble_approx_below_power_of_2(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    std::span<const QubitId> Q_target,
    const stride_span_z &modulus,
    double btol) {
    auto mark = builder.raii_mark_block_entry("idouble_approx_below_power_of_2");

    size_t modulus_cutoff = modulus.size();
    while (modulus_cutoff > 0 && modulus[modulus_cutoff - 1] == true) {
        modulus_cutoff--;
    }
    if (modulus_cutoff + btol > modulus.size()) {
        std::stringstream ss;
        ss << "modulus_cutoff=" << modulus_cutoff << " + btol=" << btol << " > modulus.size()=" << modulus.size();
        throw std::invalid_argument(ss.str());
    }
    throw_unless(modulus[0] == true, "modulus must be odd (bottom QubitOrBitOrBool set to true; not a Bit)");

    size_t bottom_span = modulus_cutoff + ceil(btol);
    throw_unless(
        ctx.clean_workspace.size() >= bottom_span,
        "not enough clean workspace for gen_idouble_approx_below_power_of_2");

    for (size_t k = Q_target.size(); k-- > 1;) {
        builder.swap(Q_target[k], Q_target[k - 1]);
    }

    gen_isub_classical(builder, ctx, Q_target.subspan(1, bottom_span - 1), modulus.skip(1), true, Q_target[0], btol);
}

void kickmix::gen_ihalve_approx_below_power_of_2(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    std::span<const QubitId> Q_target,
    const stride_span_z &modulus,
    double btol) {
    auto mark = builder.raii_mark_block_entry("ihalve_approx_below_power_of_2");

    size_t modulus_cutoff = modulus.size();
    while (modulus_cutoff > 0 && modulus[modulus_cutoff - 1] == true) {
        modulus_cutoff--;
    }
    throw_unless(modulus_cutoff + btol <= modulus.size(), "modulus_cutoff + btol > modulus.size()");
    throw_unless(modulus[0] == true, "modulus must be odd (bottom QubitOrBitOrBool set to true; not a Bit)");

    size_t bottom_span = modulus_cutoff + ceil(btol);
    throw_unless(
        ctx.clean_workspace.size() >= bottom_span, "not enough clean workspace for gen_ihalve_approx_below_power_of_2");

    gen_iadd_classical(builder, ctx, Q_target.subspan(1, bottom_span - 1), modulus.skip(1), true, Q_target[0], btol);

    for (size_t k = 1; k < Q_target.size(); k++) {
        builder.swap(Q_target[k], Q_target[k - 1]);
    }
}
