#ifndef _KICKMIX_GEN_GF_IMUL_X_H
#define _KICKMIX_GEN_GF_IMUL_X_H

#include "kickmix/build/circuit_builder.h"
#include "kickmix/build/circuit_gen_ctx.h"
#include "kickmix/util/gf2_field.h"

namespace kickmix {

/// Multiplies a GF(2^m) register by x^k, in place.
///
/// As pseudocode:
///     Q_target = Q_target * x^k  (mod the field polynomial)
///
/// Multiplying by x is a shift of the coefficients plus, when a coefficient falls off the top, the
/// addition of the reduction polynomial. Rather than physically rotating the register once per
/// factor of x, the rotations are accumulated and applied as a single cyclic permutation at the
/// end. That turns the cost from k*(m-1) swaps into (m - gcd(m, k)) swaps, which matters a great
/// deal for the multiplier, whose hot path shifts by k = m.
///
/// When k is large enough that emitting a reduction per step would cost more than a general linear
/// map, the equivalent constant-multiplication matrix is synthesized instead.
///
/// Args:
///     builder: The builder to append operations into.
///     ctx: Preferences and resources for the operation to use.
///     field: The field the register holds an element of.
///     Q_target: The register to multiply. Its size must equal the field degree.
///     k: The power of x to multiply by. Note that, unlike a rotation amount, this may not be
///         reduced modulo m: x^m is not the identity.
void gen_gf_imul_x(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const QubitId> Q_target,
    size_t k);

/// Divides a GF(2^m) register by x^k, in place. This is the exact inverse of gen_gf_imul_x.
///
/// As pseudocode:
///     Q_target = Q_target / x^k  (mod the field polynomial)
void gen_gf_idiv_x(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const QubitId> Q_target,
    size_t k);

}  // namespace kickmix

#endif
