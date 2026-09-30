#ifndef KICKGEN_GEN_CMP_H
#define KICKGEN_GEN_CMP_H

#include "kickmix/build/circuit_builder.h"

namespace kickmix {

void gen_flip_if_lt(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    stride_span<const QubitId> Q_lhs,
    const stride_span_z &rhs,
    QubitOrMinusState Q_out,
    QubitOrBitOrBool or_equal,
    QubitOrTrue control,
    double btol);

void gen_flip_if_le(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    stride_span<const QubitId> Q_lhs,
    const stride_span_z &rhs,
    QubitOrMinusState Q_out,
    QubitOrTrue control,
    double btol);

void gen_flip_if_gt(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    stride_span<const QubitId> Q_lhs,
    const stride_span_z &rhs,
    QubitOrMinusState Q_out,
    QubitOrBitOrBool or_equal,
    QubitOrTrue control,
    double btol);

void gen_flip_if_ge(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    stride_span<const QubitId> Q_lhs,
    const stride_span_z &rhs,
    QubitOrMinusState Q_out,
    QubitOrTrue control,
    double btol);

void gen_flip_if_eq(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    stride_span<const QubitId> Q_lhs,
    const stride_span_z &rhs,
    QubitOrMinusState Q_out,
    QubitOrTrue control);

}  // namespace kickmix

#endif
