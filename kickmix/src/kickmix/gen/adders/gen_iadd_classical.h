#ifndef KICKMIX_INV_IADD_CLASSICAL_H
#define KICKMIX_INV_IADD_CLASSICAL_H

#include <span>

#include "kickmix/build/circuit_builder.h"

namespace kickmix {

/// Performs an inplace addition of a classical offset into a quantum register.
///
/// As pseudocode:
///     if control: Q_target += offset = carry_in
///
/// Args:
///     out: The builder to append operations into.
///     ctx: Context specifying available ancilla qubits.
///     Q_target: The quantum register to add into.
///     offset: The classical value to add into the quantum register.
///     carry_in: Carry input for the addition (causes an extra increment when true).
///     control: The addition only occurs conditioned on the control being true.
void gen_iadd_classical(
    CircuitBuilder &out,
    CircuitGenCtx ctx,
    stride_span<const QubitId> Q_target,
    const stride_span_z &offset,
    QubitOrBitOrBool carry_in,
    QubitOrTrue control,
    double btol);

/// Same as gen_iadd_classical, but conjugated by X gates on the target.
void gen_isub_classical(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    stride_span<const QubitId> target,
    const stride_span_z &offset,
    QubitOrBitOrBool borrow_in,
    QubitOrTrue control,
    double btol);

void gen_ixor_carries_from_addition(
    CircuitBuilder &out,
    const CircuitGenCtx &ctx,
    stride_span<const QubitId> Q_src,
    const stride_span_z &offset,
    stride_span<const QubitId> Q_dst,
    QubitOrBitOrBool carry_in,
    QubitOrTrue control);

void gen_iadd_classical_using_2clean_but_with_vented_carries(
    CircuitBuilder &out,
    const CircuitGenCtx &ctx,
    stride_span<const QubitId> Q_target,
    const stride_span_z &offset,
    QubitOrBitOrBool carry_in,
    stride_span<const QubitId> Q_carry_xor_target,
    stride_span<const BitId> vent_keys,
    QubitOrTrue control);

void gen_iadd_classical_simple(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    stride_span<const QubitId> target,
    const stride_span_z &offset,
    QubitOrBitOrBool carry_in,
    QubitOrTrue control);

}  // namespace kickmix

#endif
