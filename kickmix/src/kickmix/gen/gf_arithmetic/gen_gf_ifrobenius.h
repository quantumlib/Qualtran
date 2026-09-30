#ifndef _KICKMIX_GEN_GF_IFROBENIUS_H
#define _KICKMIX_GEN_GF_IFROBENIUS_H

#include "kickmix/build/circuit_builder.h"
#include "kickmix/build/circuit_gen_ctx.h"
#include "kickmix/util/gf2_field.h"

namespace kickmix {

/// Raises a GF(2^m) register to the power 2^k, in place.
///
/// As pseudocode:
///     Q_target = Q_target^(2^k)  (mod the field polynomial)
///
/// Over a binary field, squaring is the Frobenius endomorphism, which is GF(2)-linear. So repeated
/// squaring is a linear map realized with CNOTs and SWAPs alone, with no ancillas and no Toffolis.
/// Because the Frobenius map has order m, k is taken modulo m.
///
/// Args:
///     builder: The builder to append operations into.
///     ctx: Preferences and resources for the operation to use.
///     field: The field the register holds an element of.
///     Q_target: The register to raise. Its size must equal the field degree.
///     k: How many times to square.
void gen_gf_ifrobenius(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const QubitId> Q_target,
    size_t k);

/// Applies the inverse of gen_gf_ifrobenius, i.e. takes the 2^k-th root.
///
/// As pseudocode:
///     Q_target = Q_target^(2^(m-k))  (mod the field polynomial)
void gen_gf_ifrobenius_adjoint(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const QubitId> Q_target,
    size_t k);

}  // namespace kickmix

#endif
