#ifndef KICKGEN_IMUL2_MOD_H
#define KICKGEN_IMUL2_MOD_H

#include "kickmix/build/circuit_builder.h"

namespace kickmix {

/// Generates gates performing `target *= 2 (mod modulus)`.
///
/// Although this method supports replacing the bits of the modulus
/// with qubits, it isn't optimized to give good costs in that case.
///
/// Args:
///     builder: Where to append the circuit operations.
///     ctx: Preferences and resources for the operation to use.
///     target: The qubits containing the value to double.
///     modulus: The bits of the modulus.
///     control: Defaults to true. Determines if the operation happens.
///
/// Requires:
///     The modulus must be odd at generation time, with no padding.
///
///         modulus,front() == true
///         modulus.back() == true
///         modulus.size() == target.size()
///
///     The target must start in, and will end in, the range [0, modulus):
///
///         target < modulus
void gen_imul2_mod(
    CircuitBuilder &builder,
    CircuitGenCtx ctx,
    stride_span<const QubitId> target,
    const stride_span_z &modulus,
    QubitOrTrue control,
    double btol);

void gen_imul2_inv_mod_with_subcmp_merged_by_inlining(
    CircuitBuilder &builder,
    CircuitGenCtx ctx,
    stride_span<const QubitId> target,
    const stride_span_z &modulus,
    QubitOrTrue control);

void gen_imul2_inv_mod(
    CircuitBuilder &builder,
    CircuitGenCtx ctx,
    stride_span<const QubitId> target,
    const stride_span_z &modulus,
    QubitOrTrue control,
    double btol);
}  // namespace kickmix

#endif
