#ifndef KICKGEN_GEN_CMP_LOW_SPACE_H
#define KICKGEN_GEN_CMP_LOW_SPACE_H

#include <span>

#include "kickmix/build/circuit_builder.h"

namespace kickmix {

void masked_phase_by_prefix_using_dirty_workspace(
    CircuitBuilder &builder, const CircuitGenCtx &ctx, stride_span<const QubitId> Q_target, const stride_span_z &masks);

void gen_xif_less_than_3anc_2ntof(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    stride_span<const QubitId> Q_lhs,
    const stride_span_z &rhs,
    QubitOrMinusState Q_out,
    QubitOrBitOrBool or_equal = false,
    QubitOrBitOrBool inverted = false,
    QubitOrTrue control = true);

}  // namespace kickmix

#endif
