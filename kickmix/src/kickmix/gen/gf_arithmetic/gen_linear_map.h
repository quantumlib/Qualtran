#ifndef _KICKMIX_GEN_GF_LINEAR_MAP_H
#define _KICKMIX_GEN_GF_LINEAR_MAP_H

#include "kickmix/build/circuit_builder.h"
#include "kickmix/build/circuit_gen_ctx.h"
#include "kickmix/util/gf2_matrix.h"

namespace kickmix {

/// Applies an invertible GF(2) linear map to a register, in place, using only CNOT and SWAP gates.
///
/// As pseudocode:
///     Q_target = matrix * Q_target
///
/// The map is realized by decomposing the matrix as P * L * U and emitting the CNOTs implied by each
/// factor. This uses no ancilla qubits and emits at most n*(n-1)/2 CNOTs per triangular factor.
///
/// Args:
///     builder: The builder to append operations into.
///     ctx: Preferences and resources for the operation to use.
///     Q_target: The register to transform. Qubit k holds coordinate k of the vector.
///     matrix: The square, invertible transformation to apply. Its size must match the register.
void gen_linear_map(
    CircuitBuilder &builder, const CircuitGenCtx &ctx, stride_span<const QubitId> Q_target, const GF2Matrix &matrix);

/// Applies the inverse of a GF(2) linear map to a register, in place.
///
/// As pseudocode:
///     Q_target = matrix^-1 * Q_target
void gen_linear_map_adjoint(
    CircuitBuilder &builder, const CircuitGenCtx &ctx, stride_span<const QubitId> Q_target, const GF2Matrix &matrix);

}  // namespace kickmix

#endif
