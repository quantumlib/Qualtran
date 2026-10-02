#include "kickmix/gen/gf_arithmetic/gen_gf_mul.h"

#include <algorithm>
#include <vector>

#include "kickmix/build/circuit_gen_ctx.h"
#include "kickmix/gen/gf_arithmetic/gen_gf_iadd.h"
#include "kickmix/gen/gf_arithmetic/gen_gf_imul_classical.h"
#include "kickmix/gen/gf_arithmetic/gen_gf_imul_x.h"

using namespace kickmix;

static void gen_gf2_poly_mul_impl(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    stride_span<const QubitId> Q_target,
    stride_span<const QubitId> Q_lhs,
    stride_span<const QubitId> Q_rhs);

/// Adds (1 + x^k) * Q_lhs * Q_rhs into Q_target, as polynomials over GF(2).
///
/// The product of the two n bit factors lands at offset k of the target, and a copy of it also
/// lands at offset 0. Rather than computing the product twice, the routine surrounds a single
/// product by CNOT ladders that pre-add and post-add overlapping windows of the target. Undoing a
/// ladder after the product has been added into the window leaves exactly the shifted copy behind.
///
/// Requires Q_target.size() >= k + 2 * n - 1, where n is the size of each factor.
static void gen_poly_mul_by_one_plus_xk(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    stride_span<const QubitId> Q_target,
    stride_span<const QubitId> Q_lhs,
    stride_span<const QubitId> Q_rhs,
    size_t k) {
    size_t n = Q_lhs.size();
    if (n == 0) {
        return;
    }
    throw_unless(Q_rhs.size() == n, "gen_gf2_poly_mul: factors have different sizes");
    throw_unless(Q_target.size() >= k + 2 * n - 1, "gen_gf2_poly_mul: target is too small");

    if (n == 1) {
        // The one bit case is a single Toffoli, plus a copy of its output down to offset 0.
        builder.cx(Q_target[k], Q_target[0]);
        builder.ccx(Q_lhs[0], Q_rhs[0], Q_target[k]);
        builder.cx(Q_target[k], Q_target[0]);
        return;
    }

    // The window [k, k + 2n - 1) is where the product goes. `l` counts the target bits above that
    // window that have to be folded down into it, and `c` counts the bits below it.
    size_t l = 2 * n >= k + 1 ? 2 * n - k - 1 : 0;
    size_t c = std::min(k, 2 * n - 1);

    // This ladder does not commute with itself when l > k (a control of one gate is the target of
    // another), so it must run in descending order here and ascending order when undone.
    builder.broadcast_cx(Q_target.subspan(2 * k, l).reversed(), Q_target.subspan(k, l).reversed());
    // This ladder does commute: its controls are all at index >= k and its targets all at index
    // < k, so the whole thing can be emitted as one vectorized broadcast.
    builder.broadcast_cx(Q_target.subspan(k, c), Q_target.keep(c));

    gen_gf2_poly_mul_impl(builder, ctx, Q_target.subspan(k, 2 * n - 1), Q_lhs, Q_rhs);

    builder.broadcast_cx(Q_target.subspan(k, c), Q_target.keep(c));
    builder.broadcast_cx(Q_target.subspan(2 * k, l), Q_target.subspan(k, l));
}

static void gen_gf2_poly_mul_impl(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    stride_span<const QubitId> Q_target,
    stride_span<const QubitId> Q_lhs,
    stride_span<const QubitId> Q_rhs) {
    size_t n = Q_lhs.size();
    if (n == 0) {
        return;
    }
    // Cheap enough to keep in the recursion: the sub-register arithmetic below is easy to get
    // subtly wrong, and a clamped subspan would otherwise silently drop product terms.
    throw_unless(Q_rhs.size() == n, "gen_gf2_poly_mul: factors have different sizes");
    throw_unless(Q_target.size() == 2 * n - 1, "gen_gf2_poly_mul: target.size() != 2 * lhs.size() - 1");
    if (n == 1) {
        builder.ccx(Q_lhs[0], Q_rhs[0], Q_target[0]);
        return;
    }

    // Split each factor as lo + x^k * hi, so that the product is
    //     lo1*lo2 + x^k * ((lo1 + hi1)*(lo2 + hi2) + lo1*lo2 + hi1*hi2) + x^(2k) * hi1*hi2
    //             = (1 + x^k)*lo1*lo2 + x^k*(lo1 + hi1)*(lo2 + hi2) + (x^k + x^(2k))*hi1*hi2
    // which is three half sized products instead of four.
    size_t k = (n + 1) / 2;

    gen_poly_mul_by_one_plus_xk(builder, ctx, Q_target.keep(3 * k - 1), Q_lhs.keep(k), Q_rhs.keep(k), k);
    gen_poly_mul_by_one_plus_xk(
        builder, ctx, Q_target.subspan(k, k + 2 * (n - k) - 1), Q_lhs.skip(k), Q_rhs.skip(k), k);

    // Temporarily replace the low halves by lo + hi, so the middle product can reuse the same
    // registers, then put them back.
    builder.broadcast_cx(Q_lhs.skip(k), Q_lhs.keep(n - k));
    builder.broadcast_cx(Q_rhs.skip(k), Q_rhs.keep(n - k));
    gen_gf2_poly_mul_impl(builder, ctx, Q_target.subspan(k, 2 * k - 1), Q_lhs.keep(k), Q_rhs.keep(k));
    builder.broadcast_cx(Q_rhs.skip(k), Q_rhs.keep(n - k));
    builder.broadcast_cx(Q_lhs.skip(k), Q_lhs.keep(n - k));
}

void kickmix::gen_gf2_poly_mul(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    stride_span<const QubitId> Q_target,
    stride_span<const QubitId> Q_lhs,
    stride_span<const QubitId> Q_rhs) {
    size_t n = Q_lhs.size();
    if (n == 0) {
        return;
    }
    throw_unless(Q_rhs.size() == n, "gen_gf2_poly_mul: factors have different sizes");
    throw_unless(Q_target.size() == 2 * n - 1, "gen_gf2_poly_mul: target.size() != 2 * lhs.size() - 1");
    gen_gf2_poly_mul_impl(builder, ctx, Q_target, Q_lhs, Q_rhs);
}

void kickmix::gen_gf_mul(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const QubitId> Q_target,
    stride_span<const QubitId> Q_lhs,
    stride_span<const QubitId> Q_rhs,
    QubitOrTrue control) {
    size_t m = field.degree();
    throw_unless(Q_target.size() == m, "gen_gf_mul: Q_target.size() != field.degree()");
    throw_unless(Q_lhs.size() == m, "gen_gf_mul: Q_lhs.size() != field.degree()");
    throw_unless(Q_rhs.size() == m, "gen_gf_mul: Q_rhs.size() != field.degree()");

    if (control.is_qubit()) {
        // Controlling every Toffoli of the product would turn them all into three control gates.
        // It is much cheaper to compute the product unconditionally into a temporary register, add
        // that in under the control, and then uncompute the temporary for free with measurements.
        auto mark = builder.raii_mark_block_entry("c_gf_mul");
        CircuitGenCtx sub_ctx = ctx;
        auto Q_temp = sub_ctx.take_clean(m, "gen_gf_mul");
        builder.broadcast_reset(Q_temp);
        gen_gf_mul(builder, sub_ctx, field, Q_temp, Q_lhs, Q_rhs);
        gen_gf_iadd(builder, sub_ctx, Q_target, Q_temp, control);
        gen_gf_unmul(builder, sub_ctx, field, Q_temp, Q_lhs, Q_rhs);
        return;
    }

    auto mark = builder.raii_mark_block_entry("gf_mul");
    if (m == 1) {
        // GF(2) multiplication is a single AND.
        builder.ccx(Q_lhs[0], Q_rhs[0], Q_target[0]);
        return;
    }

    size_t k = (m + 1) / 2;
    // 1 + x^k, the factor that the Karatsuba identity attaches to the outer two subproducts.
    GF2Poly one_plus_xk = field.one() ^ GF2Poly::monomial(k);

    // Reading the register as R, the sequence below maps
    //     R -> ((R/x^k + mid) / (1 + x^k) + hi) * x^k + lo, then all times (1 + x^k)
    //        = R + (1 + x^k)*lo + x^k*mid + (x^k + x^(2k))*hi
    //        = R + lhs*rhs
    // with every intermediate staying inside the m qubit register, because the shifts and the
    // constant multiplications reduce modulo the field polynomial as they go.
    gen_gf_idiv_x(builder, ctx, field, Q_target, k);

    builder.broadcast_cx(Q_lhs.skip(k), Q_lhs.keep(m - k));
    builder.broadcast_cx(Q_rhs.skip(k), Q_rhs.keep(m - k));
    gen_gf2_poly_mul_impl(builder, ctx, Q_target.keep(2 * k - 1), Q_lhs.keep(k), Q_rhs.keep(k));
    builder.broadcast_cx(Q_rhs.skip(k), Q_rhs.keep(m - k));
    builder.broadcast_cx(Q_lhs.skip(k), Q_lhs.keep(m - k));

    gen_gf_idiv_classical(builder, ctx, field, Q_target, one_plus_xk);

    gen_gf2_poly_mul_impl(builder, ctx, Q_target.keep(2 * (m - k) - 1), Q_lhs.skip(k), Q_rhs.skip(k));

    gen_gf_imul_x(builder, ctx, field, Q_target, k);

    gen_gf2_poly_mul_impl(builder, ctx, Q_target.keep(2 * k - 1), Q_lhs.keep(k), Q_rhs.keep(k));

    gen_gf_imul_classical(builder, ctx, field, Q_target, one_plus_xk);
}

void kickmix::gen_gf_phase_by_product(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const BitId> B_mask,
    stride_span<const QubitId> Q_lhs,
    stride_span<const QubitId> Q_rhs) {
    (void)ctx;
    size_t m = field.degree();
    throw_unless(B_mask.size() == m, "gen_gf_phase_by_product: B_mask.size() != field.degree()");
    throw_unless(Q_lhs.size() == m, "gen_gf_phase_by_product: Q_lhs.size() != field.degree()");
    throw_unless(Q_rhs.size() == m, "gen_gf_phase_by_product: Q_rhs.size() != field.degree()");

    auto mark = builder.raii_mark_block_entry("gf_phase_by_product");

    // Write the unreduced product as low + x^m * high, where
    //     low[i]  = sum over j <= i of lhs[j] * rhs[i - j]
    //     high[i] = sum over j > i of lhs[m - j + i] * rhs[j]
    // The low half pairs up with the mask bit of the same index.
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j <= i; j++) {
            builder.cz_if(Q_lhs[j], Q_rhs[i - j], B_mask[i]);
        }
    }

    // The high half only reaches the mask after being multiplied by x^m modulo the field
    // polynomial. Pushing that matrix onto the mask instead of onto the product means folding, for
    // each column, the handful of mask bits the column selects into a single parity bit.
    GF2Poly x_to_the_m = field.mod(GF2Poly::monomial(m));
    GF2Matrix fold = field.constant_mul_matrix(x_to_the_m);
    std::vector<size_t> column;
    for (size_t i = 0; i + 1 < m; i++) {
        column.clear();
        for (size_t r = 0; r < m; r++) {
            if (fold.get(r, i)) {
                column.push_back(r);
            }
        }
        if (column.empty()) {
            continue;
        }
        if (column.size() == 1) {
            // A weight one column needs no parity bit at all.
            for (size_t j = i + 1; j < m; j++) {
                builder.cz_if(Q_lhs[m - j + i], Q_rhs[j], B_mask[column[0]]);
            }
            continue;
        }
        auto parity = builder.alloc_clean_raii_bit();
        for (size_t r : column) {
            builder.cx(B_mask[r], parity.bit);
        }
        for (size_t j = i + 1; j < m; j++) {
            builder.cz_if(Q_lhs[m - j + i], Q_rhs[j], parity.bit);
        }
        // Restore the parity bit to zero, so that the next allocation really is clean.
        for (size_t r : column) {
            builder.cx(B_mask[r], parity.bit);
        }
    }
}

void kickmix::gen_gf_unmul(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const QubitId> Q_target,
    stride_span<const QubitId> Q_lhs,
    stride_span<const QubitId> Q_rhs,
    QubitOrTrue control) {
    size_t m = field.degree();
    throw_unless(Q_target.size() == m, "gen_gf_unmul: Q_target.size() != field.degree()");
    throw_unless(Q_lhs.size() == m, "gen_gf_unmul: Q_lhs.size() != field.degree()");
    throw_unless(Q_rhs.size() == m, "gen_gf_unmul: Q_rhs.size() != field.degree()");

    if (control.is_qubit()) {
        // Under a control the target is not known to hold the product, so the measurement trick
        // does not apply. Adding the product again is the inverse, since addition is XOR.
        gen_gf_mul(builder, ctx, field, Q_target, Q_lhs, Q_rhs, control);
        return;
    }

    auto mark = builder.raii_mark_block_entry("gf_unmul");

    // X basis measurement collapses each target qubit to zero while leaking one bit of phase
    // information, which the conditioned CZs below pay back.
    std::vector<CircuitBuilderRaiiXBit> measured;
    std::vector<BitId> B_mask;
    measured.reserve(m);
    B_mask.reserve(m);
    for (size_t i = 0; i < m; i++) {
        measured.push_back(builder.hmr_raii_xbit(Q_target[i]));
        B_mask.push_back(measured.back().bit);
    }

    gen_gf_phase_by_product(builder, ctx, field, B_mask, Q_lhs, Q_rhs);
}
