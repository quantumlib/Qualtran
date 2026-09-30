#ifndef KICKGEN_GEN_IADD1_H
#define KICKGEN_GEN_IADD1_H

#include "kickmix/build/circuit_builder.h"

namespace kickmix {

void gen_iadd1(
    CircuitBuilder &builder, CircuitGenCtx ctx, stride_span<const QubitId> Q_target, QubitOrTrue control = true);

}  // namespace kickmix

#endif
