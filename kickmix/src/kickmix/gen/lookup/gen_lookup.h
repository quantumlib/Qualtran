#ifndef KICKGEN_GEN_LOOKUP_H
#define KICKGEN_GEN_LOOKUP_H

#include <span>

#include "kickmix/build/circuit_builder.h"

namespace kickmix {

void gen_binary_to_unary(CircuitBuilder &builder, CircuitGenCtx ctx, std::span<const QubitId> target);
void gen_unary_to_binary(CircuitBuilder &builder, CircuitGenCtx ctx, std::span<const QubitId> target);

void gen_unlookup(
    CircuitBuilder &builder,
    CircuitGenCtx ctx,
    const stride_span_z &table_bits,
    stride_span<const QubitId> address,
    stride_span<const QubitId> output);
void gen_lookup(
    CircuitBuilder &builder,
    CircuitGenCtx ctx,
    const stride_span_z &table_bits,
    stride_span<const QubitId> address,
    stride_span<const QubitId> output,
    char pauli_type = 'X',
    QubitOrTrue control = true);

}  // namespace kickmix

#endif
