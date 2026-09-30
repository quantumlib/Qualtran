#ifndef _KICKMIX_GEN_GF_IMUL_CLASSICAL_H
#define _KICKMIX_GEN_GF_IMUL_CLASSICAL_H

#include "kickmix/build/circuit_builder.h"
#include "kickmix/build/circuit_gen_ctx.h"
#include "kickmix/util/gf2_field.h"

namespace kickmix {

/// Multiplies a GF(2^m) register by a classically known constant, in place.
///
/// As pseudocode:
///     Q_target = Q_target * constant  (mod the field polynomial)
///
/// Multiplication by a fixed element is a GF(2)-linear map, so it is realized entirely with CNOTs
/// and SWAPs and needs no ancillas. The constant must be non-zero, since multiplying by zero is not
/// reversible.
///
/// Args:
///     builder: The builder to append operations into.
///     ctx: Preferences and resources for the operation to use.
///     field: The field the register holds an element of.
///     Q_target: The register to multiply. Its size must equal the field degree.
///     constant: The non-zero field element to multiply by.
void gen_gf_imul_classical(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const QubitId> Q_target,
    const GF2Poly &constant);

/// Divides a GF(2^m) register by a classically known constant, in place.
///
/// As pseudocode:
///     Q_target = Q_target / constant  (mod the field polynomial)
///
/// This is the exact inverse of gen_gf_imul_classical.
void gen_gf_idiv_classical(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const QubitId> Q_target,
    const GF2Poly &constant);

}  // namespace kickmix

#endif
