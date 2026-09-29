#include "gen_cmp_low_space.h"

#include <functional>
#include <span>

using namespace kickmix;

/// Equivalent to `zif(masks[k] & all(Q_target[:k+1])` for each k.
///
/// Phase flips states where the first k qubits are all True, for each
/// k, with each phase flip controlled by a specified mask value.
///
/// The computation is done via toggle detection, which costs 4n
/// Toffolis but only requires dirty workspace. If 3 qubits of
/// clean workspace are available, iter_apply_suffix_controlled_actions
/// is twice as Toffoli efficient.
void kickmix::masked_phase_by_prefix_using_dirty_workspace(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    stride_span<const QubitId> Q_target,
    const stride_span_z &masks) {
    auto mark = builder.raii_mark_block_entry("masked_phase_by_prefix_using_dirty_workspace");

    size_t n = std::min(Q_target.size(), masks.size());
    throw_unless(
        ctx.dirty_workspace.count() + ctx.clean_workspace.size() + 3 >= n,
        "masked_phase_by_prefix_using_dirty_workspace: need more ancilla qubits");

    if (n == 0) {
        return;
    } else if (n == 1) {
        builder.cz(Q_target[0], masks[0]);
        return;
    } else if (n == 2) {
        builder.cz(Q_target[0], masks[0]);
        builder.ccz(Q_target[0], Q_target[1], masks[1]);
        return;
    } else if (n == 3) {
        builder.cz(Q_target[0], masks[0]);
        builder.ccz(Q_target[0], Q_target[1], masks[1]);
        builder.cccz(Q_target[0], Q_target[1], Q_target[2], masks[2], ctx.clean_workspace_if_not_minimizing());
        return;
    }

    const auto &Q_clean = ctx.clean_workspace;
    std::vector<QubitId> anc;
    anc.insert(anc.end(), Q_clean.begin(), Q_clean.end());
    ctx.dirty_workspace.copy_n_items_into(ctx.dirty_workspace.count(), anc);

    builder.cz(Q_target[0], masks[0]);
    for (size_t k = 0; k < std::min(n - 3, Q_clean.size()); k++) {
        builder.reset(anc[k]);
    }
    for (size_t step = 0; step < 2; step++) {
        for (size_t k = n - 4; k--;) {
            if (step == 0 && k < Q_clean.size()) {
                // Do nothing.
            } else if (step == 1 && k + 1 < Q_clean.size()) {
                builder.ccx(Q_target[k + 2], anc[k], builder.hmr_raii_xbit(anc[k + 1]));
            } else {
                builder.ccx(Q_target[k + 2], anc[k], anc[k + 1]);
            }
        }

        if (step == 1 and Q_clean.size() > 0) {
            builder.ccx(Q_target[0], Q_target[1], builder.hmr_raii_xbit(anc[0]));
        } else {
            builder.ccx(Q_target[0], Q_target[1], anc[0]);
        }

        if (step == 0) {
            for (size_t k = 0; k < std::min(Q_clean.size(), n - 4); k++) {
                builder.ccx(Q_target[k + 2], anc[k], anc[k + 1]);
            }
        }
        for (size_t k = Q_clean.size(); k < n - 4; k++) {
            builder.ccx(Q_target[k + 2], anc[k], anc[k + 1]);
        }

        for (size_t k = 0; k < n - 3; k++) {
            if (k >= Q_clean.size() || step == 0) {
                builder.cz(masks[k + 1], anc[k]);
            }
        }
        if (Q_clean.size() < n - 3 || step == 0) {
            builder.ccz(masks[n - 2], Q_target[n - 2], anc[n - 4]);
            builder.cccz(masks[n - 1], Q_target[n - 1], Q_target[n - 2], anc[n - 4]);
        }
    }
}

static void gen_flip_if_lt_fast_using_only_3clean(
    CircuitBuilder &builder,
    CircuitGenCtx ctx,
    stride_span<const QubitId> lhs,
    const stride_span_z &rhs,
    QubitOrMinusState out,
    QubitOrBitOrBool or_equal) {
    if (!rhs.is_all_classical()) {
        throw std::invalid_argument("Called gen_flip_if_lt_fast_using_only_3clean with qubit rhs.");
    }

    if (lhs.size() < rhs.size()) {
        throw std::invalid_argument("lhs.size() < rhs.size()");
    }
    if (lhs.size() > rhs.size()) {
        auto extended_rhs = array_z::copy_of_concat(rhs, stride_span_z::repeat_false(lhs.size() - rhs.size()));
        gen_flip_if_lt_fast_using_only_3clean(builder, ctx, lhs, extended_rhs, out, or_equal);
        return;
    }

    auto clean = ctx.take_clean(3, "low workspace comparison");
    QubitId a = clean[0];
    QubitId b = clean[1];
    QubitId c = clean[2];

    size_t n = lhs.size();
    if (n == 0) {
        builder.cx(or_equal, out);
        return;
    }
    if (n == 1) {
        builder.cx(rhs[0], lhs[0]);
        builder.parity_ccx({rhs[0], or_equal}, lhs[0], out);
        builder.cx(rhs[0], lhs[0]);
        builder.cx(or_equal, out);
        return;
    }
    if (n == 2) {
        builder.x(lhs[0]);
        builder.x(lhs[1]);
        builder.cx(or_equal, lhs[0]);
        builder.reset(clean[0]);
        builder.parity_ccx({or_equal, rhs[0]}, lhs[0], clean[0]);
        builder.cx(or_equal, clean[0]);
        builder.cx(clean[0], lhs[1]);
        builder.cx(rhs[1], clean[0]);
        builder.ccx(lhs[1], clean[0], out);
        builder.cx(rhs[1], clean[0]);
        builder.cx(clean[0], lhs[1]);
        builder.cx(clean[0], out);
        {
            auto raii_pushed = builder.hmr_raii_push_condition(clean[0]);
            builder.cz(lhs[0], rhs[0]);
            builder.cz(lhs[0], or_equal);
            builder.z(or_equal);
        }
        builder.cx(or_equal, lhs[0]);
        builder.x(lhs[0]);
        builder.x(lhs[1]);
        return;
    }

    builder.broadcast_x(lhs);
    builder.broadcast_cx(rhs, lhs);

    builder.cx(rhs[n - 1], out);
    builder.parity_ccx({rhs[n - 1], rhs[n - 2]}, lhs[n - 1], out);
    builder.reset(a);
    builder.reset(c);
    builder.ccx(lhs[n - 1], lhs[n - 2], a);
    builder.cx(a, c);

    auto pass =
        [&](size_t offset, size_t run, size_t d1, BitIdOrFalse mm0, BitId mm1, QubitId qa, QubitId qb, bool is_last) {
            builder.broadcast_x(lhs.subspan(offset + 1, run));
            builder.for_each(0, std::min(run, offset), [&](LoopBuilder &loop, iota k) {
                loop.ccx(lhs[k + offset], lhs[-k + offset - 1], lhs[k + offset + 1]);
            });
            if (d1 > 0) {
                builder.reset(qb);
                builder.ccx(qa, lhs[n - d1], qb);
                if (mm0.is_bit()) {
                    builder.hmr(qa, mm0.bit());
                }
            }
            if (is_last) {
                builder.cccx(or_equal, qb, lhs[offset * 2], out);
            }
            builder.reset(qa);
            builder.cx(rhs[offset + 1], qa);
            builder.cx(rhs[offset], qa);
            if (rhs.common_type == QXZTypeTag8::BIT_ID) {
                auto rhs_b = rhs.cast_data<BitId>();
                builder.for_each(0, std::min(run, offset), [&](LoopBuilder &loop, iota k) {
                    loop.cx_if(lhs[k + offset], qa, rhs_b[-k + offset]);
                    loop.cx_if(lhs[k + offset], qa, rhs_b[-k + offset - 1]);
                });
            } else {
                for (size_t k = 0; k < run && k < offset; k++) {
                    builder.ccx(lhs[offset + k], rhs[offset - k], qa);
                    builder.ccx(lhs[offset + k], rhs[offset - k - 1], qa);
                }
            }
            if (offset <= run) {
                builder.ccx(lhs[offset + offset], rhs[0], qa);
            }
            builder.ccx(qa, qb, out);
            builder.hmr(qa, mm1);
            builder.push_condition(mm1);
            builder.z(rhs[offset + 1]);
            builder.z(rhs[offset]);
            if (rhs.common_type == QXZTypeTag8::BIT_ID) {
                auto rhs_b = rhs.cast_data<BitId>();
                builder.for_each(0, std::min(run, offset), [&](LoopBuilder &loop, iota k) {
                    loop.z_if(lhs[k + offset], rhs_b[-k + offset]);
                    loop.z_if(lhs[k + offset], rhs_b[-k + offset - 1]);
                });
            } else {
                for (size_t k = 0; k < run && k < offset; k++) {
                    builder.cz(lhs[offset + k], rhs[offset - k]);
                    builder.cz(lhs[offset + k], rhs[offset - k - 1]);
                }
            }
            builder.pop_condition();
            if (offset <= run) {
                builder.ccz(lhs[offset + offset], rhs[0], mm1);
            }
        };
    std::vector<CircuitBuilderRaiiBit> ms;
    size_t nest = 0;
    while (true) {
        ms.push_back(builder.alloc_dirty_raii_bit());
        size_t shift = (2 << nest) + nest + 1;
        size_t next_shift = (4 << nest) + nest + 2;
        bool is_last = next_shift > n;
        if (shift > n) {
            break;
        }
        pass(
            n - shift,
            2 << nest,
            nest,
            nest == 0 ? BitIdOrFalse{} : BitIdOrFalse(ms[nest - 1].bit),
            ms[nest].bit,
            nest % 2 == 0 ? b : a,
            nest % 2 == 0 ? a : b,
            is_last);
        nest++;
    }

    builder.hmr(nest % 2 == 0 ? b : a, ms[nest - 1].bit);
    {
        std::vector<QubitId> prefix_targets{c};
        std::vector<QubitOrBitOrBool> prefix_masks;
        for (size_t k = n; k-- > n - nest + 1;) {
            prefix_targets.push_back(lhs[k]);
        }
        for (auto &m : ms) {
            prefix_masks.push_back(m.bit);
        }
        auto dirty = lhs.subspan(0, std::max(size_t{2}, nest - 1) - 2);
        masked_phase_by_prefix_using_dirty_workspace(
            builder, CircuitGenCtx{clean.subspan(0, 2)}.with_more_dirty_qubits(dirty), prefix_targets, prefix_masks);
    }
    builder.hmr(c, ms.back().bit);
    auto inv_pass = [&builder, &lhs](size_t offset, size_t run) {
        builder.for_each_reversed(0, std::min(offset, run), [&](LoopBuilder &loop, iota k) {
            loop.ccx(lhs[k + offset], lhs[-k + offset - 1], lhs[k + offset + 1]);
        });
        builder.broadcast_x(lhs.subspan(offset + 1, run));
    };
    while (nest--) {
        inv_pass(n - (2 << nest) - nest - 1, 2 << nest);
    }
    builder.ccz(lhs[n - 1], lhs[n - 2], ms.back().bit);
    builder.broadcast_x(lhs);
    builder.broadcast_cx(rhs, lhs);
}
void kickmix::gen_xif_less_than_3anc_2ntof(
    CircuitBuilder &out,
    const CircuitGenCtx &ctx,
    stride_span<const QubitId> Q_lhs,
    const stride_span_z &rhs,
    QubitOrMinusState Q_out,
    QubitOrBitOrBool or_equal,
    QubitOrBitOrBool inverted,
    QubitOrTrue control) {
    auto mark = out.raii_mark_block_entry("xif_less_than_3anc_2ntof");

    out.ccx(inverted, control, Q_out);
    if (control.is_qubit()) {
        std::vector<QubitId> clhs;
        for (auto e : Q_lhs) {
            clhs.push_back(e);
        }
        clhs.push_back((QubitId)control);
        out.x((QubitId)control);
        gen_flip_if_lt_fast_using_only_3clean(out, ctx, clhs, rhs, Q_out, or_equal);
        out.x((QubitId)control);
    } else {
        gen_flip_if_lt_fast_using_only_3clean(out, ctx, Q_lhs, rhs, Q_out, or_equal);
    }
}
