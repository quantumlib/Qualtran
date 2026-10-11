#include "gen_qft_demolition_measure.h"

using namespace kickmix;

void kickmix::gen_qft_demolition_measure(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const stride_span<const QubitId> &target,
    const stride_span<const BitId> &output,
    bool inverse_qft) {
    if (target.size() != output.size()) {
        throw std::invalid_argument("target.size() != output.size()");
    }
    auto mark = builder.raii_mark_block_entry("qft_demolition_measure");

    size_t n = target.size();
    builder.broadcast_swap(target.keep(n / 2), target.keep_last(n / 2).reversed());

    for (size_t k = 0; k < n; k++) {
        auto q = target[k];
        for (size_t k2 = 0; k2 < k; k2++) {
            auto angle = FixedPrecisionAngle128::from_power_of_2_half_turns((int)k2 - (int)k);
            if (inverse_qft) {
                angle = -angle;
            }
            builder.z_pow_if(q, angle, output[k2]);
        }
        builder.hmr(q, output[k]);
    }
}
