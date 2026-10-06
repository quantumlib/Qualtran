#include "kickmix/gen/gf_arithmetic/gen_gf_inverse.h"

#include <bit>
#include <vector>

#include "kickmix/build/circuit_gen_ctx.h"
#include "kickmix/gen/gf_arithmetic/gen_gf_iadd.h"
#include "kickmix/gen/gf_arithmetic/gen_gf_ifrobenius.h"
#include "kickmix/gen/gf_arithmetic/gen_gf_mul.h"

using namespace kickmix;

namespace {

/// The shape of the binary addition chain that reaches the exponent m - 1.
struct InverseChain {
    /// The field degree.
    size_t m;
    /// popcount(m - 1), i.e. how many doubled values have to be folded together at the end.
    size_t t;
    /// floor(log2(m - 1)), i.e. how many doubling steps the chain performs.
    size_t k1;
    /// Index of the output register within the chain's register list. The list has k + 1 entries:
    /// the input at index 0, then k - 1 workspace registers, then the output.
    size_t k;
    /// Positions of the set bits of m - 1, from most significant to least significant.
    std::vector<size_t> set_bits;
};

InverseChain plan_inverse_chain(size_t m, bool has_explicit_chain) {
    InverseChain chain;
    chain.m = m;
    chain.t = std::popcount(m - 1);
    chain.k1 = std::bit_width(m - 1) - 1;
    if (chain.t == 1) {
        // A power of two exponent needs no folding step, so the doubling chain is the whole story.
        chain.k = chain.k1 + 1;
    } else if (has_explicit_chain) {
        // The last fold can write straight into the output register instead of into workspace.
        chain.k = chain.k1 + chain.t - 1;
    } else {
        chain.k = chain.k1 + chain.t;
    }
    for (size_t i = 0; i <= chain.k1; i++) {
        if (((m - 1) >> (chain.k1 - i)) & 1) {
            chain.set_bits.push_back(chain.k1 - i);
        }
    }
    return chain;
}

/// Lays out the chain's registers: the input at index 0, then the chain registers, then the output.
std::vector<stride_span<const QubitId>> layout_chain_registers(
    const InverseChain &chain,
    stride_span<const QubitId> Q_target,
    stride_span<const QubitId> Q_input,
    stride_span<const QubitId> Q_chain) {
    std::vector<stride_span<const QubitId>> f;
    f.reserve(chain.k + 1);
    f.push_back(Q_input);
    for (size_t i = 1; i < chain.k; i++) {
        f.push_back(Q_chain.subspan((i - 1) * chain.m, chain.m));
    }
    f.push_back(Q_target);
    return f;
}

}  // namespace

size_t kickmix::gf_inverse_workspace_size(const GF2Field &field) {
    size_t m = field.degree();
    if (m == 1) {
        return 0;
    }
    return m * (plan_inverse_chain(m, false).k - 1);
}

size_t kickmix::gf_inverse_chain_size(const GF2Field &field) {
    size_t m = field.degree();
    if (m == 1) {
        return 0;
    }
    return m * (plan_inverse_chain(m, true).k - 1);
}

void kickmix::gen_gf_inverse(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const QubitId> Q_target,
    stride_span<const QubitId> Q_input,
    stride_span<const QubitId> Q_chain) {
    size_t m = field.degree();
    throw_unless(Q_target.size() == m, "gen_gf_inverse: Q_target.size() != field.degree()");
    throw_unless(Q_input.size() == m, "gen_gf_inverse: Q_input.size() != field.degree()");
    throw_unless(
        Q_chain.size() == gf_inverse_chain_size(field),
        "gen_gf_inverse: Q_chain.size() != gf_inverse_chain_size(field)");

    auto mark = builder.raii_mark_block_entry("gf_inverse_chain");
    if (m == 1) {
        // Both elements of GF(2) are their own inverse.
        gen_gf_iadd(builder, ctx, Q_target, Q_input);
        return;
    }

    InverseChain chain = plan_inverse_chain(m, true);
    size_t k = chain.k;
    size_t k1 = chain.k1;
    size_t t = chain.t;
    const std::vector<size_t> &set_bits = chain.set_bits;
    auto f = layout_chain_registers(chain, Q_target, Q_input, Q_chain);

    // Doubling steps: f[i] = beta(2^i), built from beta(2^(i-1))^(2^(2^(i-1))) * beta(2^(i-1)).
    // The output register is borrowed as scratch for the Frobenius-shifted copy, since it is still
    // zero at this point.
    for (size_t i = 1; i <= k1; i++) {
        gen_gf_iadd(builder, ctx, f[k], f[i - 1]);
        gen_gf_ifrobenius(builder, ctx, field, f[k], size_t{1} << (i - 1));
        gen_gf_mul(builder, ctx, field, f[i], f[i - 1], f[k]);
        gen_gf_ifrobenius_adjoint(builder, ctx, field, f[k], size_t{1} << (i - 1));
        gen_gf_iadd(builder, ctx, f[k], f[i - 1]);
    }

    // Folding steps: accumulate one set bit of m - 1 at a time, most significant first. The left
    // operand is left Frobenius-shifted rather than shifted back, which the cleanup undoes.
    for (size_t s = 1; s < t; s++) {
        gen_gf_ifrobenius(builder, ctx, field, f[k1 + s - 1], size_t{1} << set_bits[s]);
        gen_gf_mul(builder, ctx, field, f[k1 + s], f[k1 + s - 1], f[set_bits[s]]);
    }

    if (t == 1) {
        // beta(m - 1) is sitting in f[k1]. Move it into the output register, leaving a zeroed
        // register behind to act as scratch for the cleanup. When k1 is zero there is no chain
        // register to move from, and f[k1] would be the input, so the value is copied instead.
        if (k1 == 0) {
            gen_gf_iadd(builder, ctx, f[k], f[0]);
        } else {
            builder.broadcast_swap(f[k1], f[k]);
        }
    }

    // beta(m - 1)^2 = x^(2^m - 2) = 1/x.
    gen_gf_ifrobenius(builder, ctx, field, f[k], 1);
}

void kickmix::gen_gf_uninverse(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const QubitId> Q_target,
    stride_span<const QubitId> Q_input,
    stride_span<const QubitId> Q_chain) {
    size_t m = field.degree();
    throw_unless(Q_target.size() == m, "gen_gf_uninverse: Q_target.size() != field.degree()");
    throw_unless(Q_input.size() == m, "gen_gf_uninverse: Q_input.size() != field.degree()");
    throw_unless(
        Q_chain.size() == gf_inverse_chain_size(field),
        "gen_gf_uninverse: Q_chain.size() != gf_inverse_chain_size(field)");

    auto mark = builder.raii_mark_block_entry("gf_uninverse_chain");
    if (m == 1) {
        // Both elements of GF(2) are their own inverse.
        gen_gf_iadd(builder, ctx, Q_target, Q_input);
        builder.broadcast_del_zero(Q_target);
        return;
    }

    InverseChain chain = plan_inverse_chain(m, true);
    size_t k = chain.k;
    size_t k1 = chain.k1;
    size_t t = chain.t;
    const std::vector<size_t> &set_bits = chain.set_bits;
    auto f = layout_chain_registers(chain, Q_target, Q_input, Q_chain);

    gen_gf_ifrobenius_adjoint(builder, ctx, field, f[k], 1);

    // Q_chain already holds the chain's intermediate values, so it only has to be unwound once,
    // with no multiplications to rebuild it first.
    if (t == 1) {
        if (k1 == 0) {
            gen_gf_iadd(builder, ctx, f[k], f[0]);
        } else {
            builder.broadcast_swap(f[k1], f[k]);
        }
    } else {
        for (size_t s = t - 1; s >= 1; s--) {
            gen_gf_unmul(builder, ctx, field, f[k1 + s], f[k1 + s - 1], f[set_bits[s]]);
            gen_gf_ifrobenius_adjoint(builder, ctx, field, f[k1 + s - 1], size_t{1} << set_bits[s]);
        }
        // s == t - 1 uncomputed f[k] (Q_target) via gen_gf_unmul, which emitted HMR on f[k].
        // Re-activate f[k] before using it as the zeroed scratch register for the doubling cleanup.
        builder.broadcast_reset(f[k]);
    }
    for (size_t i = k1; i >= 1; i--) {
        gen_gf_iadd(builder, ctx, f[k], f[i - 1]);
        gen_gf_ifrobenius(builder, ctx, field, f[k], size_t{1} << (i - 1));
        gen_gf_unmul(builder, ctx, field, f[i], f[i - 1], f[k]);
        gen_gf_ifrobenius_adjoint(builder, ctx, field, f[k], size_t{1} << (i - 1));
        gen_gf_iadd(builder, ctx, f[k], f[i - 1]);
    }
    builder.broadcast_del_zero(Q_target);
}

void kickmix::gen_gf_inverse(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const QubitId> Q_target,
    stride_span<const QubitId> Q_input) {
    size_t m = field.degree();
    throw_unless(Q_target.size() == m, "gen_gf_inverse: Q_target.size() != field.degree()");
    throw_unless(Q_input.size() == m, "gen_gf_inverse: Q_input.size() != field.degree()");

    auto mark = builder.raii_mark_block_entry("gf_inverse");
    if (m == 1) {
        // Both elements of GF(2) are their own inverse.
        gen_gf_iadd(builder, ctx, Q_target, Q_input);
        return;
    }

    InverseChain chain = plan_inverse_chain(m, false);
    size_t k = chain.k;
    size_t k1 = chain.k1;
    size_t t = chain.t;
    const std::vector<size_t> &set_bits = chain.set_bits;
    CircuitGenCtx sub_ctx = ctx;
    auto workspace = sub_ctx.take_clean(m * (k - 1), "gen_gf_inverse");
    builder.broadcast_reset(workspace);
    auto f = layout_chain_registers(chain, Q_target, Q_input, workspace);

    // Doubling steps: f[i] = beta(2^i), built from beta(2^(i-1))^(2^(2^(i-1))) * beta(2^(i-1)).
    // The output register is borrowed as scratch for the Frobenius-shifted copy, since it is still
    // zero at this point.
    for (size_t i = 1; i <= k1; i++) {
        gen_gf_iadd(builder, sub_ctx, f[k], f[i - 1]);
        gen_gf_ifrobenius(builder, sub_ctx, field, f[k], size_t{1} << (i - 1));
        gen_gf_mul(builder, sub_ctx, field, f[i], f[i - 1], f[k]);
        gen_gf_ifrobenius_adjoint(builder, sub_ctx, field, f[k], size_t{1} << (i - 1));
        gen_gf_iadd(builder, sub_ctx, f[k], f[i - 1]);
    }

    // Folding steps: accumulate one set bit of m - 1 at a time, most significant first. The left
    // operand is left Frobenius-shifted rather than shifted back, which the cleanup undoes.
    for (size_t s = 1; s < t; s++) {
        gen_gf_ifrobenius(builder, sub_ctx, field, f[k1 + s - 1], size_t{1} << set_bits[s]);
        gen_gf_mul(builder, sub_ctx, field, f[k1 + s], f[k1 + s - 1], f[set_bits[s]]);
    }

    if (t == 1) {
        // beta(m - 1) is sitting in f[k1]. Move it into the output register, leaving a zeroed
        // register behind to act as scratch for the cleanup. When k1 is zero there is no chain
        // register to move from, and f[k1] would be the input, so the value is copied instead.
        if (k1 == 0) {
            gen_gf_iadd(builder, sub_ctx, f[k], f[0]);
        } else {
            builder.broadcast_swap(f[k1], f[k]);
            // Undo the doubling steps. The swap above left f[k1] holding zero, and nothing below
            // writes to it, so it serves as the scratch register for every step.
            for (size_t i = k1 - 1; i >= 1; i--) {
                gen_gf_iadd(builder, sub_ctx, f[k1], f[i - 1]);
                gen_gf_ifrobenius(builder, sub_ctx, field, f[k1], size_t{1} << (i - 1));
                gen_gf_unmul(builder, sub_ctx, field, f[i], f[i - 1], f[k1]);
                gen_gf_ifrobenius_adjoint(builder, sub_ctx, field, f[k1], size_t{1} << (i - 1));
                gen_gf_iadd(builder, sub_ctx, f[k1], f[i - 1]);
            }
            builder.broadcast_del_zero(f[k1]);
        }
    } else {
        // Copy beta(m - 1) out of the last workspace register, then run the whole chain backwards.
        gen_gf_iadd(builder, sub_ctx, f[k], f[k - 1]);
        for (size_t s = t - 1; s >= 1; s--) {
            gen_gf_unmul(builder, sub_ctx, field, f[k1 + s], f[k1 + s - 1], f[set_bits[s]]);
            gen_gf_ifrobenius_adjoint(builder, sub_ctx, field, f[k1 + s - 1], size_t{1} << set_bits[s]);
        }
        // The fold cleanup just measured f[k1 + 1] out, so it has to be reset before being reused
        // as the scratch register that undoes the doubling steps.
        builder.broadcast_reset(f[k1 + 1]);
        for (size_t i = k1; i >= 1; i--) {
            gen_gf_iadd(builder, sub_ctx, f[k1 + 1], f[i - 1]);
            gen_gf_ifrobenius(builder, sub_ctx, field, f[k1 + 1], size_t{1} << (i - 1));
            gen_gf_unmul(builder, sub_ctx, field, f[i], f[i - 1], f[k1 + 1]);
            gen_gf_ifrobenius_adjoint(builder, sub_ctx, field, f[k1 + 1], size_t{1} << (i - 1));
            gen_gf_iadd(builder, sub_ctx, f[k1 + 1], f[i - 1]);
        }
        builder.broadcast_del_zero(f[k1 + 1]);
    }

    // beta(m - 1)^2 = x^(2^m - 2) = 1/x.
    gen_gf_ifrobenius(builder, sub_ctx, field, f[k], 1);
}

void kickmix::gen_gf_uninverse(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const QubitId> Q_target,
    stride_span<const QubitId> Q_input) {
    size_t m = field.degree();
    throw_unless(Q_target.size() == m, "gen_gf_uninverse: Q_target.size() != field.degree()");
    throw_unless(Q_input.size() == m, "gen_gf_uninverse: Q_input.size() != field.degree()");

    auto mark = builder.raii_mark_block_entry("gf_uninverse");
    if (m == 1) {
        // Both elements of GF(2) are their own inverse.
        gen_gf_iadd(builder, ctx, Q_target, Q_input);
        builder.broadcast_del_zero(Q_target);
        return;
    }

    InverseChain chain = plan_inverse_chain(m, false);
    size_t k = chain.k;
    size_t k1 = chain.k1;
    size_t t = chain.t;
    const std::vector<size_t> &set_bits = chain.set_bits;
    CircuitGenCtx sub_ctx = ctx;
    auto workspace = sub_ctx.take_clean(m * (k - 1), "gen_gf_uninverse");
    builder.broadcast_reset(workspace);
    auto f = layout_chain_registers(chain, Q_target, Q_input, workspace);

    gen_gf_ifrobenius_adjoint(builder, sub_ctx, field, f[k], 1);

    // With a clean workspace there is nothing to unwind yet, so the chain has to be rebuilt before
    // it can be taken apart from the other end.
    if (t == 1) {
        for (size_t i = 1; i + 1 <= k1; i++) {
            gen_gf_iadd(builder, sub_ctx, f[i + 1], f[i - 1]);
            gen_gf_ifrobenius(builder, sub_ctx, field, f[i + 1], size_t{1} << (i - 1));
            gen_gf_mul(builder, sub_ctx, field, f[i], f[i - 1], f[i + 1]);
            gen_gf_ifrobenius_adjoint(builder, sub_ctx, field, f[i + 1], size_t{1} << (i - 1));
            gen_gf_iadd(builder, sub_ctx, f[i + 1], f[i - 1]);
        }
        if (k1 == 0) {
            gen_gf_iadd(builder, sub_ctx, f[k], f[0]);
        } else {
            builder.broadcast_swap(f[k1], f[k]);
        }
    } else {
        for (size_t i = 1; i <= k1; i++) {
            gen_gf_iadd(builder, sub_ctx, f[i + 1], f[i - 1]);
            gen_gf_ifrobenius(builder, sub_ctx, field, f[i + 1], size_t{1} << (i - 1));
            gen_gf_mul(builder, sub_ctx, field, f[i], f[i - 1], f[i + 1]);
            gen_gf_ifrobenius_adjoint(builder, sub_ctx, field, f[i + 1], size_t{1} << (i - 1));
            gen_gf_iadd(builder, sub_ctx, f[i + 1], f[i - 1]);
        }
        for (size_t s = 1; s < t; s++) {
            gen_gf_ifrobenius(builder, sub_ctx, field, f[k1 + s - 1], size_t{1} << set_bits[s]);
            gen_gf_mul(builder, sub_ctx, field, f[k1 + s], f[k1 + s - 1], f[set_bits[s]]);
        }
        gen_gf_iadd(builder, sub_ctx, f[k], f[k - 1]);
    }

    for (size_t s = t - 1; s >= 1; s--) {
        gen_gf_unmul(builder, sub_ctx, field, f[k1 + s], f[k1 + s - 1], f[set_bits[s]]);
        gen_gf_ifrobenius_adjoint(builder, sub_ctx, field, f[k1 + s - 1], size_t{1} << set_bits[s]);
    }
    for (size_t i = k1; i >= 1; i--) {
        gen_gf_iadd(builder, sub_ctx, f[k], f[i - 1]);
        gen_gf_ifrobenius(builder, sub_ctx, field, f[k], size_t{1} << (i - 1));
        gen_gf_unmul(builder, sub_ctx, field, f[i], f[i - 1], f[k]);
        gen_gf_ifrobenius_adjoint(builder, sub_ctx, field, f[k], size_t{1} << (i - 1));
        gen_gf_iadd(builder, sub_ctx, f[k], f[i - 1]);
    }
    builder.broadcast_del_zero(Q_target);
}
