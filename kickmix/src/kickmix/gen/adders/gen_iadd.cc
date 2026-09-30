#include "gen_iadd.h"

#include <span>

using namespace kickmix;

void kickmix::gen_iadd(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    stride_span<const QubitId> target,
    stride_span<const QubitId> offset,
    QubitOrTrue control) {
    auto mark = builder.raii_mark_block_entry("iadd");

    const size_t n = target.size();
    const size_t m = offset.size();
    if (n == 0 || m == 0) {
        return;
    } else if (n == 1) {
        builder.ccx(control, offset[0], target[0]);
        return;
    } else if (n == 2) {
        builder.cccx(control, offset[0], target[0], target[1], ctx.clean_workspace_if_not_minimizing());
        builder.ccx(control, offset[0], target[0]);
        if (m > 1) {
            builder.ccx(control, offset[1], target[1]);
        }
        return;
    } else if (n <= m) {
        // Offset can be truncated.
        // DIDNTDO: Cancel this CX against a CX in the recursion.
        builder.ccx(control, offset[n - 1], target[n - 1]);
        gen_iadd(builder, ctx.with_more_dirty_qubits(offset.subspan(n - 1)), target, offset.subspan(0, n - 1), control);
        return;
    }
    if (n > m + 1) {
        size_t padding_qubits = n - (m + 1);
        if (ctx.clean_workspace.size() < padding_qubits) {
            throw std::invalid_argument(
                "Not implemented: target.size() > offset.size() + ctx.clean_workspace.size() + "
                "!ctx.dirty_workspace.empty()");
        }
        std::vector<QubitId> extended_offset;
        for (auto e : offset) {
            extended_offset.push_back(e);
        }
        for (size_t k = 0; k < padding_qubits; k++) {
            extended_offset.push_back(ctx.clean_workspace[k]);
        }
        gen_iadd(builder, ctx.with_clean_subspan(padding_qubits), target, extended_offset, control);
        return;
    }
    QubitId carry;
    bool clean_carry;
    if (!ctx.clean_workspace.empty()) {
        carry = ctx.clean_workspace.front();
        builder.reset(carry);
        clean_carry = true;
    } else if (!ctx.dirty_workspace.empty()) {
        carry = ctx.dirty_workspace.back();
        clean_carry = false;
    } else {
        throw std::invalid_argument(
            "Not implemented: target.size() > offset.size() + ctx.clean_workspace.size() + "
            "!ctx.dirty_workspace.empty()");
    }
    throw_unless(n == m + 1, "Failed to match offset size to target size.");

    if (!clean_carry) {
        builder.broadcast_cx(carry, target.subspan(1));
        builder.broadcast_cx(carry, offset.subspan(1));
    }

    size_t clean_stop = ctx.minimize_qubits ? 1 : ctx.clean_workspace.size();
    clean_stop = std::max(std::min(clean_stop, n - 2), size_t{1});

    {
        builder.ccx(target[0], offset[0], carry);
        builder.cx(carry, offset[1]);
        builder.cx(carry, target[1]);

        auto clean = stride_span<const QubitId>(ctx.clean_workspace);
        builder.for_each(1, clean_stop, [&](LoopBuilder &loop, iota k) {
            loop.reset(clean[k]);
            loop.ccx(target[k], offset[k], clean[k]);
            loop.cx(clean[k], carry);
            loop.cx(carry, offset[k + 1]);
            loop.cx(carry, target[k + 1]);
        });

        builder.for_each(clean_stop, n - 2, [&](LoopBuilder &loop, iota k) {
            loop.ccx(target[k], offset[k], carry);
            loop.cx(carry, offset[k + 1]);
            loop.cx(carry, target[k + 1]);
        });
    }
    builder.ccx(control, carry, target[n - 1]);
    builder.cccx(control, target[n - 2], offset[n - 2], target[n - 1]);
    {
        builder.for_each_reversed(clean_stop, n - 2, [&](LoopBuilder &loop, iota k) {
            loop.mux_ccx(control, offset[k + 1], target[k + 1]);
            loop.cx(carry, target[k + 1]);
            loop.cx(carry, offset[k + 1]);
            loop.ccx(target[k], offset[k], carry);
        });

        {
            auto mx = builder.alloc_dirty_raii_bit();
            stride_ptr<const QubitId> cleanx = {.ptr = ctx.clean_workspace.data(), .stride = 1};
            builder.for_each_reversed(1, clean_stop, [&](LoopBuilder &loop, iota k) {
                loop.mux_ccx(control, offset[k + 1], target[k + 1]);
                loop.cx(carry, target[k + 1]);
                loop.cx(carry, offset[k + 1]);
                loop.cx(cleanx, carry);
                loop.hmr(cleanx, mx.bit);
                loop.cz_if(target[k], offset[k], mx.bit);
            });
        }

        builder.ccx(control, offset[1], target[1]);
        builder.cx(carry, target[1]);
        builder.cx(carry, offset[1]);
        if (clean_carry) {
            builder.del_and(target[0], offset[0], carry);
        } else {
            builder.ccx(target[0], offset[0], carry);
        }
    }

    builder.ccx(control, offset[0], target[0]);
    if (!clean_carry) {
        builder.broadcast_cx(carry, offset.subspan(1));
        builder.broadcast_cx(carry, target.subspan(1));
        builder.ccx(control, carry, target[n - 1]);
    }
}

void kickmix::gen_isub(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    stride_span<const QubitId> target,
    stride_span<const QubitId> offset,
    QubitOrTrue control) {
    auto mark = builder.raii_mark_block_entry("isub");
    builder.broadcast_x(target);
    gen_iadd(builder, ctx, target, offset, control);
    builder.broadcast_x(target);
}
