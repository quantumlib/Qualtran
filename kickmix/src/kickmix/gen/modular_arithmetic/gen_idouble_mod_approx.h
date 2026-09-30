#ifndef KICKGEN_GEN_IDOUBLE_APPROX_H
#define KICKGEN_GEN_IDOUBLE_APPROX_H

#include <span>

#include "kickmix/build/circuit_builder.h"

namespace kickmix {

void gen_idouble_approx_below_power_of_2(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    std::span<const QubitId> Q_target,
    const stride_span_z &modulus,
    double btol);
void gen_ihalve_approx_below_power_of_2(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    std::span<const QubitId> Q_target,
    const stride_span_z &modulus,
    double btol);

}  // namespace kickmix

#endif
