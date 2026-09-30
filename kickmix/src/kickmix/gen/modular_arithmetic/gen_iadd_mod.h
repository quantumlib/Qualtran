#ifndef KICKGEN_GEN_IADD_MOD_H
#define KICKGEN_GEN_IADD_MOD_H

#include <span>

#include "kickmix/build/circuit_builder.h"

namespace kickmix {

/// Performs `target += offset (mod modulus)`.
void gen_iadd_mod(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    std::span<const QubitId> target,
    std::span<const QubitId> offset,
    const stride_span_z &modulus,
    QubitOrTrue control,
    double btol);

/// Performs `target -= offset (mod modulus)`.
void gen_isub_mod(
    CircuitBuilder &builder,
    CircuitGenCtx ctx,
    std::span<const QubitId> target,
    std::span<const QubitId> offset,
    const stride_span_z &modulus,
    QubitOrTrue control,
    double btol);

}  // namespace kickmix

#endif
