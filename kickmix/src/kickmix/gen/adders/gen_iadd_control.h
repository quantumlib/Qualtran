#ifndef KICKGEN_GEN_IADD_CONTROL_H
#define KICKGEN_GEN_IADD_CONTROL_H

#include "kickmix/build/circuit_builder.h"

namespace kickmix {

/// Performs `if control: Q_target += Q_offset + carry_in`.
void gen_iadd_control(
    CircuitBuilder &builder,
    CircuitGenCtx ctx,
    stride_span<const QubitId> Q_target,
    stride_span<const QubitId> Q_offset,
    QubitOrBitOrBool carry_in,
    QubitId control);

/// Performs `if control: Q_target -= Q_offset`.
void gen_isub_control(
    CircuitBuilder &builder,
    CircuitGenCtx ctx,
    stride_span<const QubitId> Q_target,
    stride_span<const QubitId> Q_offset,
    QubitId control);

}  // namespace kickmix

#endif
