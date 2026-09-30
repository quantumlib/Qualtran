#ifndef _KICKMIX_GEN_GF_IADD_H
#define _KICKMIX_GEN_GF_IADD_H

#include "kickmix/build/circuit_builder.h"
#include "kickmix/build/circuit_gen_ctx.h"

namespace kickmix {

/// Adds one GF(2^m) register into another, in place.
///
/// Addition in a binary extension field is bitwise XOR, so this is a layer of CNOTs (or Toffolis
/// when controlled). No reduction is needed because the sum of two elements of degree less than m
/// again has degree less than m.
///
/// As pseudocode:
///     if control: Q_target ^= Q_offset
///
/// Args:
///     builder: The builder to append operations into.
///     ctx: Preferences and resources for the operation to use.
///     Q_target: The register to add into. Its contents are replaced.
///     Q_offset: The register to add. Its contents are left alone.
///     control: The addition only occurs conditioned on the control being true.
///
/// Requires:
///     Q_target and Q_offset have the same size and are disjoint, and the control is not one of
///     their qubits.
///     Register disjointness is a precondition, not something the generators verify. Scanning
///     the registers on every call would cost more than the gates being emitted, so passing
///     overlapping registers silently produces wrong arithmetic rather than an error.
void gen_gf_iadd(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    stride_span<const QubitId> Q_target,
    const stride_span_z &offset,
    QubitOrTrue control = true);

}  // namespace kickmix

#endif
