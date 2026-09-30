#ifndef KICKGEN_INEG_MOD_H
#define KICKGEN_INEG_MOD_H

#include <span>

#include "kickmix/build/circuit_builder.h"

namespace kickmix {

/// Generates gates performing `target *= -1 (mod modulus)`.
///
/// Although this method supports replacing the bits of the modulus
/// with qubits, it isn't optimized to give good costs in that case.
///
/// Args:
///     builder: Where to append the circuit operations.
///     ctx: Preferences and resources for the operation to use.
///     target: The qubits containing the value to double.
///     modulus: The bits of the modulus.
///
/// Requires:
///     The target must start in, and will end in, the range [0, modulus):
///
///         target < modulus
void gen_ineg_mod(
    CircuitBuilder &builder, CircuitGenCtx ctx, std::span<const QubitId> target, const stride_span_z &modulus);
}  // namespace kickmix

#endif
