#ifndef KICKGEN_GEN_QFT_DEMOLITION_MEASURE_H
#define KICKGEN_GEN_QFT_DEMOLITION_MEASURE_H

#include <span>

#include "kickmix/build/circuit_builder.h"

namespace kickmix {

/// Performs a demolition frequency basis measurement.
void gen_qft_demolition_measure(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const stride_span<const QubitId> &target,
    const stride_span<const BitId> &output,
    bool inverse_qft);

}  // namespace kickmix

#endif
