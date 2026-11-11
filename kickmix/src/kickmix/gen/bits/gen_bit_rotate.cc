#include "gen_bit_rotate.h"

using namespace kickmix;

static void controlled_reverse(
    CircuitBuilder &builder, const stride_span<const QubitId> &target, QubitOrTrue control = true) {
    size_t n = target.size();
    size_t h = n >> 1;
    auto v1 = target.subspan(0, h);
    auto v2 = target.reversed().subspan(0, h);
    builder.broadcast_cswap(control, v2, v1);
}

static void controlled_shifted_reverse(
    CircuitBuilder &builder, const stride_span<const QubitId> &target, int64_t offset, QubitOrTrue control = true) {
    if (target.empty()) {
        return;
    }
    size_t n = target.size();
    offset %= (int64_t)n;
    offset += n;
    offset %= (int64_t)n;
    controlled_reverse(builder, target.keep((size_t)offset), control);
    controlled_reverse(builder, target.skip((size_t)offset), control);
}

struct RaiiStoreXor {
    CircuitBuilder *builder;
    QubitOrBitOrBool v1;
    QubitOrBitOrBool v2;
    QubitOrBitOrBool result;
    QubitOrBitOrBool apply() {
        if (v1.is_qubit()) {
            if (v2.is_qubit()) {
                builder->cx((QubitId)v1, (QubitId)v2);
                return v2;
            } else if (v2.is_bit()) {
                builder->cx((BitId)v2, (QubitId)v1);
                return v1;
            } else {
                builder->cx((bool)v2, (QubitId)v1);
                return v1;
            }
        } else if (v1.is_bit()) {
            if (v2.is_qubit()) {
                builder->cx((BitId)v1, (QubitId)v2);
                return v2;
            } else if (v2.is_bit()) {
                builder->cx((BitId)v1, (BitId)v2);
                return v2;
            } else {
                builder->cx((bool)v2, (BitId)v1);
                return v1;
            }
        } else {
            if (v2.is_qubit()) {
                builder->cx((bool)v1, (QubitId)v2);
                return v2;
            } else if (v2.is_bit()) {
                builder->cx((bool)v1, (BitId)v2);
                return v2;
            } else {
                return (bool)v1 != (bool)v2;
            }
        }
    }
    RaiiStoreXor(CircuitBuilder &builder, QubitOrBitOrBool v1, QubitOrBitOrBool v2)
        : builder(&builder), v1(v1), v2(v2) {
        result = apply();
    }
    ~RaiiStoreXor() {
        apply();
    }
};

template <typename TCallback>
static void if_and_do(
    CircuitBuilder &self, const CircuitGenCtx &ctx, QubitOrBitOrBool v1, QubitOrBitOrBool v2, TCallback callback) {
    if (v2 == false || v1 == false) {
        return;
    }
    if (v1.is_bit()) {
        self.push_condition((BitId)v1);
    }
    if (v2.is_bit()) {
        self.push_condition((BitId)v2);
    }
    QubitOrTrue control = true;
    if (v1.is_qubit() && v2.is_qubit()) {
        if (ctx.clean_workspace.empty()) {
            std::stringstream ss;
            ss << "Need 1 clean qubit to temporarily store ";
            ss << v1 << " & " << v2 << ", but no clean qubits are available.";
            throw std::invalid_argument(ss.str());
        }
        control = ctx.clean_workspace[0];
        self.reset((QubitId)control);
        self.ccx((QubitId)v1, (QubitId)v2, (QubitId)control);
    } else if (v1.is_qubit()) {
        control = (QubitId)v1;
    } else if (v2.is_qubit()) {
        control = (QubitId)v2;
    }
    callback(control);
    if (v1.is_qubit() && v2.is_qubit()) {
        self.del_and((QubitId)v1, (QubitId)v2, (QubitId)control);
    }
    if (v1.is_bit()) {
        self.pop_condition();
    }
    if (v2.is_bit()) {
        self.pop_condition();
    }
}

void kickmix::gen_left_rotate(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const stride_span<const QubitId> &target,
    int64_t shift,
    QubitOrTrue control,
    std::string_view mark_name) {
    auto mark = builder.raii_mark_block_entry(mark_name);
    if (target.empty()) {
        return;
    }
    shift %= (int64_t)target.size();
    shift += target.size();
    shift %= target.size();
    if (shift == 0) {
        return;
    }
    controlled_shifted_reverse(builder, target, 0, control);
    controlled_shifted_reverse(builder, target, shift, control);
}

void kickmix::gen_left_rotate(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const stride_span<const QubitId> &target,
    const stride_span_z &shift,
    QubitOrTrue control,
    bool inverted,
    std::string_view mark_name) {
    auto mark = builder.raii_mark_block_entry(mark_name);
    if (shift.size() > 60) {
        throw std::invalid_argument("len(shift) > 60");
    }
    for (size_t k = 0; k <= shift.size(); k++) {
        QubitOrBitOrBool c0 = k > 0 ? shift[k - 1] : false;
        QubitOrBitOrBool c1 = k < shift.size() ? shift[k] : false;
        RaiiStoreXor parity(builder, c0, c1);
        if_and_do(builder, ctx, parity.result, control, [&](QubitOrTrue merged_control) {
            controlled_shifted_reverse(builder, target, (int64_t{inverted ? -1 : +1} << k) + 1, merged_control);
        });
    }
}

void kickmix::gen_right_rotate(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const stride_span<const QubitId> &target,
    int64_t shift,
    QubitOrTrue control,
    std::string_view mark_name) {
    auto mark = builder.raii_mark_block_entry(mark_name);
    gen_left_rotate(builder, ctx, target, -shift, control, "");
}

void kickmix::gen_right_rotate(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const stride_span<const QubitId> &target,
    const stride_span_z &shift,
    QubitOrTrue control,
    bool inverted,
    std::string_view mark_name) {
    auto mark = builder.raii_mark_block_entry(mark_name);
    gen_left_rotate(builder, ctx, target, shift, control, !inverted, "");
}
