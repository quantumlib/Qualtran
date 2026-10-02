#include "gen_iadd_classical.h"

#include <iostream>
#include <span>

#include "gen_iadd.h"

using namespace kickmix;

static constexpr QubitId UNUSED_QUBIT = QubitId(UINT32_MAX & UNTAGGED_MASK);

/// A circuit for `Q_dst ^= carry(Q_src, offset, carry_in) >> 1`.
void kickmix::gen_ixor_carries_from_addition(
    CircuitBuilder &out,
    const CircuitGenCtx &ctx,
    stride_span<const QubitId> Q_src,
    const stride_span_z &offset,
    stride_span<const QubitId> Q_dst,
    QubitOrBitOrBool carry_in,
    QubitOrTrue control) {
    if (Q_src.size() < Q_dst.size()) {
        throw std::invalid_argument("Q_src.size() < Q_dst.size()");
    }

    auto mark = out.raii_mark_block_entry(control.is_qubit() ? "c_ixor_carries" : "ixor_carries");
    Q_src = Q_src.subspan(0, Q_dst.size());
    size_t n = Q_src.size();

    QubitOrBitOrBool off0 = offset.size() > 0 ? offset[0] : false;
    if (n == 0) {
        return;
    }
    if (n == 1) {
        out.cx(carry_in, Q_src[0]);
        out.parity_cccx({off0, carry_in}, control, Q_src[0], Q_dst[0], ctx.clean_workspace);
        out.cx(carry_in, Q_src[0]);
        out.ccx(control, carry_in, Q_dst[0]);
        return;
    }

    out.broadcast_ccx(control, offset, Q_src);

    out.for_each_reversed(0, Q_dst.size() - 1, [&](LoopBuilder &loop, iota k) {
        loop.ccx(Q_src[k + 1], Q_dst[k], Q_dst[k + 1]);
    });

    out.broadcast_ccx(control, offset, Q_dst);
    out.broadcast_ccx(control, offset.skip(1), Q_dst);
    out.parity_cccx({carry_in, off0}, control, Q_src[0], Q_dst[0], ctx.clean_workspace);

    out.for_each(0, Q_dst.size() - 1, [&](LoopBuilder &loop, iota k) {
        loop.ccx(Q_src[k + 1], Q_dst[k], Q_dst[k + 1]);
    });

    out.broadcast_ccx(control, offset.skip(1), Q_dst);
    out.broadcast_ccx(control, offset, Q_src);
}

void kickmix::gen_iadd_classical_using_2clean_but_with_vented_carries(
    CircuitBuilder &out,
    const CircuitGenCtx &ctx,
    stride_span<const QubitId> Q_target,
    const stride_span_z &offset,
    QubitOrBitOrBool carry_in,
    stride_span<const QubitId> Q_carry_xor_target,
    stride_span<const BitId> vent_keys,
    QubitOrTrue control) {
    auto mark = out.raii_mark_block_entry(control.is_qubit() ? "c_iadd_classical_vented" : "iadd_classical_vented");

    if (Q_carry_xor_target.size() > Q_target.size()) {
        std::vector<QubitId> new_targets;
        std::vector<QubitId> new_xor_targets;
        for (size_t k = 0; k < Q_target.size(); k++) {
            new_xor_targets.push_back(Q_carry_xor_target[k]);
            new_targets.push_back(Q_target[k]);
        }
        new_targets.push_back(Q_carry_xor_target[Q_target.size()]);
        gen_iadd_classical_using_2clean_but_with_vented_carries(
            out, ctx, new_targets, offset, carry_in, new_xor_targets, vent_keys, control);
        return;
    }
    if (offset.size() < Q_target.size()) {
        gen_iadd_classical_using_2clean_but_with_vented_carries(
            out,
            ctx,
            Q_target,
            array_z::copy_of_concat(offset, stride_span_z::repeat_false(Q_target.size() - offset.size())),
            carry_in,
            Q_carry_xor_target,
            vent_keys,
            control);
        return;
    }

    if (vent_keys.size() + 2 < Q_target.size()) {
        throw std::invalid_argument("gen_iadd_classical_using_2clean_but_with_vented_carries: not enough vent keys");
    }
    if (ctx.clean_workspace.size() < 2) {
        throw std::invalid_argument("gen_iadd_classical_using_2clean_but_with_vented_carries: need 2 clean qubits");
    }
    for (size_t k = 1; k < Q_carry_xor_target.size(); k++) {
        if (Q_carry_xor_target[k] == UNUSED_QUBIT) {
            throw std::invalid_argument("Q_carry_xor_target contained a NO_QUBIT past its first entry.");
        }
    }
    bool carry_xor_target_0_is_no_qubit = Q_carry_xor_target.size() > 0 && Q_carry_xor_target[0] == UNUSED_QUBIT;
    bool has_carry_xor_target_0 = Q_carry_xor_target.size() > 0 && Q_carry_xor_target[0] != UNUSED_QUBIT;

    size_t n = Q_target.size();

    if (n == 0) {
        return;
    }
    if (n == 1) {
        out.ccx(control, offset[0], Q_target[0]);
        if (has_carry_xor_target_0) {
            out.cx(Q_target[0], Q_carry_xor_target[0]);
        }
        out.ccx(control, carry_in, Q_target[0]);
        if (has_carry_xor_target_0) {
            out.cx(Q_target[0], Q_carry_xor_target[0]);
        }
        return;
    }
    if (n == 2) {
        out.broadcast_ccx(control, offset, Q_target);
        out.broadcast_cx(
            Q_target.keep(Q_carry_xor_target.size()).skip(carry_xor_target_0_is_no_qubit),
            Q_carry_xor_target.skip(carry_xor_target_0_is_no_qubit));
        out.parity_cccx({offset[0], carry_in}, control, Q_target[0], Q_target[1], ctx.clean_workspace);
        out.ccx(control, offset[0], Q_target[1]);
        out.ccx(control, carry_in, Q_target[0]);
        out.broadcast_cx(
            Q_target.keep(Q_carry_xor_target.size()).skip(carry_xor_target_0_is_no_qubit),
            Q_carry_xor_target.skip(carry_xor_target_0_is_no_qubit));
        return;
    }
    if (n == 3) {
        out.broadcast_ccx(control, offset, Q_target);
        out.broadcast_cx(
            Q_target.keep(Q_carry_xor_target.size()).skip(carry_xor_target_0_is_no_qubit),
            Q_carry_xor_target.skip(carry_xor_target_0_is_no_qubit));

        QubitId c0 = ctx.clean_workspace[0];
        out.parity_cccx({offset[0], carry_in}, control, Q_target[0], c0, ctx.clean_workspace.subspan(1));
        out.ccx(control, offset[0], c0);
        out.cx(c0, Q_target[1]);
        out.ccx(control, offset[1], c0);
        out.ccx(c0, Q_target[1], Q_target[2]);
        out.cx(c0, Q_target[2]);
        out.ccx(c0, offset[1], Q_target[2]);
        out.ccx(control, offset[1], Q_target[2]);

        auto mx = out.alloc_dirty_raii_bit();
        out.hmr(c0, mx.bit);
        out.parity_cccz({offset[0], carry_in}, control, Q_target[0], mx.bit, {});
        out.ccz(control, offset[0], mx.bit);
        out.ccz(control, offset[1], mx.bit);
        out.ccx(control, carry_in, Q_target[0]);

        out.broadcast_cx(
            Q_target.keep(Q_carry_xor_target.size()).skip(carry_xor_target_0_is_no_qubit),
            Q_carry_xor_target.skip(carry_xor_target_0_is_no_qubit));
        return;
    }

    QubitId c0 = ctx.clean_workspace[0];
    QubitId c1 = ctx.clean_workspace[1];

    out.broadcast_ccx(control, offset, Q_target);
    out.broadcast_cx(
        Q_target.keep(Q_carry_xor_target.size()).skip(carry_xor_target_0_is_no_qubit),
        Q_carry_xor_target.skip(carry_xor_target_0_is_no_qubit));

    out.reset(c0);
    out.parity_cccx({offset[0], carry_in}, control, Q_target[0], c0, {});
    out.ccx(control, carry_in, Q_target[0]);
    out.ccx(control, offset[0], c0);

    auto mx = out.alloc_dirty_raii_bit();
    for (size_t k = 1; k < n - 3; k++) {
        out.reset(c1);
        out.swap(c0, c1);
        out.ccx(control, offset[k], c1);
        out.ccx(Q_target[k], c1, c0);
        out.ccx(control, offset[k], c1);
        out.cx(c1, Q_target[k]);
        out.hmr(c1, mx.bit);
        out.bit_invert_if(vent_keys[k], mx.bit);
        out.ccx(control, offset[k], c0);
    }

    out.reset(c1);
    out.ccx(control, offset[n - 3], c0);
    out.ccx(Q_target[n - 3], c0, c1);
    out.ccx(control, offset[n - 3], c0);
    out.cx(c0, Q_target[n - 3]);
    out.ccx(control, offset[n - 3], c1);
    out.ccx(control, offset[n - 2], c1);
    out.ccx(Q_target[n - 2], c1, Q_target[n - 1]);
    out.ccx(control, offset[n - 2], c1);
    out.cx(c1, Q_target[n - 2]);
    out.hmr(c1, mx.bit);

    out.push_condition(mx.bit);

    out.ccz(offset[n - 3], Q_target[n - 3], control);
    out.ccz(offset[n - 3], c0, control);
    out.cz(offset[n - 3], control);

    out.cz(Q_target[n - 3], c0);
    out.z(c0);
    out.pop_condition();

    out.ccx(control, offset[n - 2], Q_target[n - 1]);
    out.hmr(c0, mx.bit);
    out.cx(mx.bit, vent_keys[n - 3]);

    out.broadcast_cx(
        Q_target.keep(Q_carry_xor_target.size()).skip(carry_xor_target_0_is_no_qubit),
        Q_carry_xor_target.skip(carry_xor_target_0_is_no_qubit));
}

void kickmix::gen_iadd_classical(
    CircuitBuilder &out,
    CircuitGenCtx ctx,
    stride_span<const QubitId> Q_target,
    const stride_span_z &offset,
    QubitOrBitOrBool carry_in,
    QubitOrTrue control,
    double btol) {
    auto mark = out.raii_mark_block_entry(control.is_qubit() ? "c_iadd_classical" : "iadd_classical");

    if (offset.size() < Q_target.size()) {
        gen_iadd_classical(
            out,
            ctx,
            Q_target,
            array_z::copy_of_concat(offset, stride_span_z::repeat_false(Q_target.size() - offset.size())),
            carry_in,
            control,
            btol);
        return;
    }

    if (Q_target.size() == 0) {
        return;
    }
    if (Q_target.size() == 1) {
        out.parity_ccx({carry_in, offset[0]}, control, Q_target[0]);
        return;
    }
    throw_unless(ctx.clean_workspace.size() >= 2, "gen_iadd_classical: need 2 clean qubits");
    if (Q_target.size() == 2) {
        auto a = ctx.clean_workspace[0];
        auto t0 = Q_target[0];
        auto t1 = Q_target[1];
        auto o0 = offset[0];
        auto o1 = offset[1];
        out.cx(o0, t0);
        out.cx(o1, t1);
        out.reset_and(t0, control, a);
        out.parity_ccx({carry_in, o0}, a, t1);
        out.del_and(t0, control, a);
        out.parity_ccx({o0, o1}, control, t1);
        out.parity_ccx({o0, carry_in}, control, t0);
        out.cx(o0, t0);
        out.cx(o1, t1);
        return;
    }
    if (Q_target.size() == 3) {
        auto a = ctx.clean_workspace[0];
        auto b = ctx.clean_workspace[1];
        auto t0 = Q_target[0];
        auto t1 = Q_target[1];
        auto t2 = Q_target[2];
        auto o0 = offset[0];
        auto o1 = offset[1];
        auto o2 = offset[2];
        out.reset(a);
        out.reset(b);
        out.cx(o0, t0);
        out.cx(o1, t1);
        out.parity_ccx({carry_in, o0}, t0, a);
        out.cx(o0, a);
        out.cx(o1, a);
        out.ccx(control, a, b);
        out.ccx(t1, b, t2);
        out.cx(b, t1);
        out.ccx(control, a, out.hmr_raii_xbit(b));
        {
            auto mxa = out.hmr_raii_xbit(a);
            out.parity_ccx({carry_in, o0}, t0, mxa);
            out.cx(o0, mxa);
            out.cx(o1, mxa);
        }
        out.cx(o0, t0);
        out.cx(o1, t1);
        out.parity_ccx({o1, o2}, control, t2);
        out.parity_ccx({o0, carry_in}, control, t0);
        return;
    }
    if (ctx.clean_workspace.size() + ctx.dirty_workspace.count() < Q_target.size()) {
        std::stringstream ss;
        ss << "gen_iadd_classical: need n=" << Q_target.size() << " ancilla qubits (at least 2 clean)\n";
        ss << "got " << ctx.clean_workspace.size() << " clean ancilla qubits\n";
        ss << "got " << ctx.dirty_workspace.count() << " dirty ancilla qubits\n";
        throw std::invalid_argument(ss.str());
    }

    // Approximate addition by truncating carries.
    if (btol < Q_target.size() && offset[Q_target.size() - 1].is_bool()) {
        btol = std::max(btol, 0.0);
        bool inverted = (bool)offset[Q_target.size() - 1];
        size_t n_affected = Q_target.size();
        while (n_affected > 0 && offset[n_affected - 1] == inverted) {
            n_affected -= 1;
        }
        n_affected += (size_t)ceil(btol);
        n_affected = std::min(n_affected, Q_target.size());
        if (n_affected < Q_target.size()) {
            stride_span<const QubitId> sub_target = Q_target.subspan(0, n_affected);
            array_z sub_offset = array_z::copy_of(offset.keep(n_affected));
            if (inverted) {
                out.inplace_invert(sub_offset);
                out.broadcast_x(sub_target);
                out.inplace_invert(carry_in);
            }
            gen_iadd_classical(
                out,
                ctx.with_more_dirty_qubits(Q_target.subspan(n_affected)),
                sub_target,
                sub_offset,
                carry_in,
                control,
                INFINITY);
            if (inverted) {
                out.inplace_invert(carry_in);
                out.broadcast_x(sub_target);
                out.inplace_invert(sub_offset);
            }
            return;
        }
    }

    // If more clean qubits are available than needed, use them to
    // save Toffolis on the bottom part of the addition.
    if (!ctx.minimize_qubits && ctx.clean_workspace.size() > 2 && Q_target.size() > 1) {
        std::span<const QubitId> Q_clean = ctx.clean_workspace;
        size_t m = std::min(Q_target.size() - 1, Q_clean.size() - 2);
        out.broadcast_reset(Q_clean.subspan(0, m));
        out.broadcast_ccx(control, offset.keep(m), Q_target.keep(m));
        out.broadcast_ccx(control, offset.keep(m), Q_clean.subspan(0, m));
        out.parity_cccx({carry_in, offset[0]}, control, Q_target[0], Q_clean[0], Q_clean.subspan(1));

        for (size_t k = 1; k < m; k++) {
            out.ccx(control, offset[k], Q_clean[k - 1]);
            out.ccx(Q_target[k], Q_clean[k - 1], Q_clean[k]);
        }

        gen_iadd_classical(
            out,
            ctx.with_clean_subspan(m).with_more_dirty_qubits(ctx.clean_workspace.subspan(0, m - 1)),
            Q_target.subspan(m),
            offset.skip(m),
            Q_clean[m - 1],
            control,
            btol);

        {
            auto mx = out.alloc_dirty_raii_bit();
            for (size_t k = m; k-- > 1;) {
                out.hmr(Q_clean[k], mx.bit);
                out.push_condition(mx.bit);
                out.cz(Q_target[k], Q_clean[k - 1]);
                out.cz(offset[k], control);
                out.pop_condition();
                out.ccx(offset[k], control, Q_clean[k - 1]);
                out.cx(Q_clean[k - 1], Q_target[k]);
            }
        }
        {
            auto raii_pushed = out.hmr_raii_push_condition(Q_clean[0]);
            out.parity_ccz({carry_in, offset[0]}, control, Q_target[0]);
            out.cz(control, offset[0]);
        }
        out.ccx(control, carry_in, Q_target[0]);
        return;
    }

    size_t n = Q_target.size();
    if (ctx.dirty_workspace.count() < n - 2) {
        std::stringstream ss;
        ss << "Need at least n-2 qubits of dirty workspace.\n";
        ss << "Got " << ctx.dirty_workspace.count() << " dirty qubits but need " << n - 2 << ".";
        throw std::invalid_argument(ss.str());
    }
    std::vector<QubitId> Q_dirty_pre_data;
    Q_dirty_pre_data.push_back(UNUSED_QUBIT);
    ctx.dirty_workspace.copy_n_items_into(n - 2, Q_dirty_pre_data);
    stride_span<const QubitId> Q_dirty_pre = Q_dirty_pre_data;
    std::vector<CircuitBuilderRaiiBit> vent_raii;
    std::vector<BitId> vent_bits_back;
    vent_raii.push_back(out.alloc_clean_raii_bit());
    while (vent_raii.size() < n - 1) {
        vent_raii.push_back(out.alloc_clean_raii_bit());
        vent_bits_back.push_back(vent_raii.back().bit);
    }
    stride_span<const BitId> vent_bits = vent_bits_back;
    gen_iadd_classical_using_2clean_but_with_vented_carries(
        out, ctx, Q_target, offset, carry_in, Q_dirty_pre, vent_bits, control);

    out.broadcast_x(Q_target);
    out.for_each(1, vent_bits.size(), [&](LoopBuilder &loop, iota k) {
        loop.z_if(Q_dirty_pre[k], vent_bits[k]);
    });
    gen_ixor_carries_from_addition(
        out,
        ctx,
        Q_target.subspan(0, Q_target.size() - 1),
        offset,
        stride_span<const QubitId>(Q_dirty_pre).subspan(1),
        carry_in,
        control);
    out.for_each(1, vent_bits.size(), [&](LoopBuilder &loop, iota k) {
        loop.z_if(Q_dirty_pre[k], vent_bits[k]);
    });
    out.broadcast_x(Q_target);
}

void kickmix::gen_iadd_classical_simple(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    stride_span<const QubitId> Q_target,
    const stride_span_z &offset,
    QubitOrBitOrBool carry_in,
    QubitOrTrue control) {
    auto mark = builder.raii_mark_block_entry(control.is_qubit() ? "c_iadd_classical_simple" : "iadd_classical_simple");

    auto safe_get_offset = [&](size_t k) -> QubitOrBitOrBool {
        if (k >= offset.size()) {
            return false;
        }
        return offset[k];
    };

    size_t n = Q_target.size();
    throw_unless(ctx.clean_workspace.size() >= n, "Not enough workspace for gen_iadd_classical");

    auto Q_clean = ctx.clean_workspace.subspan(0, n);
    if (n == 0) {
        return;
    }
    if (n == 1) {
        builder.ccx(carry_in, control, Q_target[0]);
        builder.ccx(offset[0], control, Q_target[0]);
        return;
    }
    if (n == 2) {
        builder.ccx(control, offset[0], Q_target[0]);
        builder.ccx(control, offset[1], Q_target[1]);
        builder.x(Q_target[0]);
        {
            builder.parity_cccx({offset[0], carry_in}, control, Q_target[0], Q_target[1], ctx.clean_workspace);
        }
        builder.x(Q_target[0]);
        builder.ccx(control, carry_in, Q_target[0]);
        builder.ccx(control, carry_in, Q_target[1]);
        return;
    }

    builder.reset(Q_clean[0]);
    builder.ccx(control, carry_in, Q_clean[0]);

    for (size_t k = 0; k < n - 1; k++) {
        builder.reset(Q_clean[k + 1]);
        builder.ccx(control, safe_get_offset(k), Q_clean[k]);
        builder.ccx(control, safe_get_offset(k), Q_target[k]);
        builder.ccx(control, safe_get_offset(k), Q_clean[k + 1]);
        builder.ccx(Q_clean[k], Q_target[k], Q_clean[k + 1]);
    }

    builder.cx(Q_clean[n - 1], Q_target[n - 1]);
    builder.ccx(control, safe_get_offset(n - 1), Q_target[n - 1]);

    {
        size_t k = n - 2;
        {
            auto mx = builder.hmr_raii_xbit(Q_clean[k + 1]);
            builder.ccx(Q_clean[k], Q_target[k], mx);
            builder.ccx(control, safe_get_offset(k), mx);
        }

        builder.cx(Q_clean[k], Q_target[k]);
        builder.ccx(control, safe_get_offset(k), Q_target[k]);
    }

    for (size_t k = n - 2; k--;) {
        {
            auto mx = builder.hmr_raii_xbit(Q_clean[k + 1]);
            builder.ccx(control, safe_get_offset(k + 1), mx);
            builder.ccx(Q_clean[k], Q_target[k], mx);
            builder.ccx(control, safe_get_offset(k), mx);
        }

        builder.cx(Q_clean[k], Q_target[k]);
        builder.ccx(control, safe_get_offset(k), Q_target[k]);
    }

    {
        auto mx = builder.hmr_raii_xbit(Q_clean[0]);
        builder.ccx(control, offset[0], mx);
        builder.ccx(control, carry_in, mx);
    }
}

void kickmix::gen_isub_classical(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    stride_span<const QubitId> target,
    const stride_span_z &offset,
    QubitOrBitOrBool borrow_in,
    QubitOrTrue control,
    double btol) {
    auto mark = builder.raii_mark_block_entry(control.is_qubit() ? "c_isub_classical" : "isub_classical");
    builder.broadcast_x(target);
    gen_iadd_classical(builder, ctx, target, offset, borrow_in, control, btol);
    builder.broadcast_x(target);
}
