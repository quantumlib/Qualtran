#ifndef KICKGEN_GEN_BIT_ROTATE_H
#define KICKGEN_GEN_BIT_ROTATE_H

#include "kickmix/build/circuit_builder.h"

namespace kickmix {

void gen_left_rotate(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const stride_span<const QubitId> &target,
    int64_t shift,
    QubitOrTrue control,
    std::string_view mark_name = "left_rotate");

void gen_left_rotate(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const stride_span<const QubitId> &target,
    const stride_span_z &shift,
    QubitOrTrue control,
    bool inverted,
    std::string_view mark_name = "left_rotate");

void gen_right_rotate(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const stride_span<const QubitId> &target,
    int64_t shift,
    QubitOrTrue control,
    std::string_view mark_name = "right_rotate");

void gen_right_rotate(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const stride_span<const QubitId> &target,
    const stride_span_z &shift,
    QubitOrTrue control,
    bool inverted,
    std::string_view mark_name = "right_rotate");

}  // namespace kickmix

#endif
