#ifndef KICKGEN_GEN_CMP_QQ_H
#define KICKGEN_GEN_CMP_QQ_H

#include "kickmix/build/circuit_builder.h"

namespace kickmix {

void gen_flip_if_lt_qq(
    CircuitBuilder &builder,
    CircuitGenCtx ctx,
    stride_span<const QubitId> lhs,
    stride_span<const QubitId> rhs,
    QubitOrMinusState out,
    QubitOrBitOrBool or_equal,
    QubitOrTrue control);

}  // namespace kickmix

#endif
