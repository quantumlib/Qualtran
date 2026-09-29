#ifndef KICKGEN_GEN_IADD_H
#define KICKGEN_GEN_IADD_H

#include "kickmix/build/circuit_builder.h"

namespace kickmix {

/// Generates gates performing `target += offset`.
///
/// Args:
///     builder: Where to append the circuit operations.
///     ctx: Preferences and resources for the operation to use.
///     target: The qubits containing the value to add into.
///     offset: The qubits containing the offset that will be
///         added into the target register.
///     control: Determines if the operation happens or not.
///
/// Requires:
///     The offset qubits are used for workspace, so the target
///     can't be much larger than the offset:
///
///         target.size() <= offset.size() + 1
void gen_iadd(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    stride_span<const QubitId> target,
    stride_span<const QubitId> offset,
    QubitOrTrue control = true);

void gen_isub(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    stride_span<const QubitId> target,
    stride_span<const QubitId> offset,
    QubitOrTrue control = true);

}  // namespace kickmix

#endif
