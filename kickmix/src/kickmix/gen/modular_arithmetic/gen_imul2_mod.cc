#include "gen_imul2_mod.h"

#include <iostream>

#include "kickmix/gen/adders/gen_iadd_classical.h"
#include "kickmix/gen/comparators/gen_cmp.h"

using namespace kickmix;

static void gen_imul2_mod_exact(
    CircuitBuilder &builder,
    CircuitGenCtx ctx,
    stride_span<const QubitId> target,
    const stride_span_z &modulus,
    QubitOrTrue control) {
    size_t n = target.size();
    throw_unless(modulus.size() > 0, "modulus.size() == 0 (division by zero will occur because modulus must be zero)");
    throw_unless(modulus.size() == n, "modulus.size() != target.size()");
    throw_unless(modulus.front() == true, "modulus must be odd (modulus[0] == true)");
    throw_unless(modulus.back() == true, "modulus must be full capacity (modulus[target.size() - 1] == true)");

    auto mark = builder.raii_mark_block_entry(control.is_qubit() ? "c_imul2_mod_exact" : "imul2_mod_exact");

    // Trivial reductions.
    if (n == 1) {
        // Only valid modulus is 1, so the value must be 0, so multiplication by 2 does nothing.
        return;
    } else if (n == 2) {
        // Only valid modulus is 3, so multiplying by 2 just exchanges 01 and 10.
        builder.cswap(control, target[0], target[1]);
        return;
    }

    stride_span<const QubitId> workspace = ctx.take_clean(n - 2);
    QubitId control_and_target0 = target[0];
    if (control.is_qubit()) {
        control_and_target0 = ctx.take_clean(1)[0];
    }

    builder.cleft_rotate(control, target);

    // Forward pass of ripple carry subtraction.
    builder.reset(workspace[0]);
    builder.x(workspace[0]);
    builder.cx(target[1], workspace[0]);
    builder.ccx(target[1], modulus[1], workspace[0]);

    {
        for (size_t k = 0; k < n - 3; k++) {
            builder.ccx(control, workspace[k], target[k + 2]);
            builder.cx(modulus[k + 2], workspace[k]);
            builder.reset(workspace[k + 1]);
            builder.ccx(target[k + 2], workspace[k], workspace[k + 1]);
            builder.cx(modulus[k + 2], workspace[k + 1]);
        }

        builder.x(workspace[n - 3]);
        builder.cccx(control, target[n - 1], workspace[n - 3], target[0], ctx.clean_workspace);
        if (control.is_qubit()) {
            builder.reset_and(control, target[0], control_and_target0);
        }
        builder.ccx(control_and_target0, workspace[n - 3], target[n - 1]);
        builder.x(workspace[n - 3]);
        builder.x(target[0]);
        if (control.is_qubit()) {
            builder.cx(control, control_and_target0);
        }

        // Reverse pass of controlled ripple carry subtraction, cancelling the subtraction on underflow.
        auto mx = builder.alloc_dirty_raii_bit();
        for (size_t k = n - 3; k--;) {
            builder.cx(modulus[k + 2], workspace[k + 1]);
            builder.hmr(workspace[k + 1], mx.bit);
            builder.cz_if(target[k + 2], workspace[k], mx.bit);
            builder.ccx(control_and_target0, workspace[k], target[k + 2]);
            builder.cx(modulus[k + 2], workspace[k]);
        }

        builder.broadcast_ccx(control, modulus.skip(2).skip_last(1), target.skip(2).skip_last(1));
    }

    {
        auto mx = builder.hmr_raii_xbit(workspace[0]);
        builder.ccx(modulus[1], target[1], mx);
        builder.cx(target[1], mx);
        builder.x(mx);
    }
    if (control.is_qubit()) {
        builder.cx(control, control_and_target0);
    }
    builder.x(target[0]);
    builder.ccx(control_and_target0, modulus[1], target[1]);
    builder.cx(control_and_target0, target[1]);
    if (control.is_qubit()) {
        builder.del_and(control, target[0], control_and_target0);
    }
}

void kickmix::gen_imul2_mod(
    CircuitBuilder &builder,
    CircuitGenCtx ctx,
    stride_span<const QubitId> target,
    const stride_span_z &modulus,
    QubitOrTrue control,
    double btol) {
    btol += 1;  // Two subroutines (approx compare and approx sub).

    size_t n = target.size();
    throw_unless(modulus.size() > 0, "modulus.size() == 0 (division by zero will occur because modulus must be zero)");
    throw_unless(modulus.size() == n, "modulus.size() != target.size()");
    throw_unless(modulus.front() == true, "modulus must be odd (modulus[0] == true)");
    throw_unless(modulus.back() == true, "modulus must be full capacity (modulus[target.size() - 1] == true)");
    if (btol >= n && ctx.clean_workspace.size() + 2 >= n && !ctx.minimize_qubits) {
        gen_imul2_mod_exact(builder, ctx, target, modulus, control);
        return;
    }
    auto mark = builder.raii_mark_block_entry(control.is_qubit() ? "c_imul2_mod_approx" : "imul2_mod_approx");
    btol = std::min(btol, (double)n);

    // Trivial reductions.
    if (target.size() == 1) {
        // Only valid modulus is 1, so the value must be 0, so multiplication by 2 does nothing.
        return;
    } else if (target.size() == 2) {
        // Only valid modulus is 3, so multiplying by 2 just exchanges 01 and 10.
        builder.cswap(control, target[0], target[1]);
        return;
    }

    array_z half_modulus = array_z::copy_of_concat(modulus.skip(1), stride_span_z::repeat_false(1));

    // Perform `anc = control and Q_target > (modulus >> 1)`.
    QubitId anc = ctx.take_clean(1)[0];
    size_t ease = 0;
    while (ease < n && modulus[n - ease - 1] == true) {
        ease += 1;
    }
    if (ease > btol) {
        // When the modulus is slightly below a power of 2, the wraparound bit is an
        // excellent predictor of the comparison.
        builder.reset_and(control, target.back(), anc);
    } else {
        gen_flip_if_gt(builder, ctx, target, half_modulus, anc, false, control, btol);
    }

    // Perform `target += ~(modulus >> 1) & ~(-1 << (n - 1))`.
    builder.inplace_invert(half_modulus);
    stride_span_z half_mod_clipped = half_modulus.as_ptr().skip_last(1);
    while (half_mod_clipped.back() == false) {
        half_mod_clipped.count--;
    }
    gen_iadd_classical(builder, ctx, target, half_mod_clipped, false, anc, btol);
    builder.inplace_invert(half_modulus);

    builder.cx(anc, target.back());
    builder.cleft_rotate(control, target);
    builder.cx(anc, target[0]);

    builder.del_and(control, target[0], anc);
}

void kickmix::gen_imul2_inv_mod_with_subcmp_merged_by_inlining(
    CircuitBuilder &builder,
    CircuitGenCtx ctx,
    stride_span<const QubitId> target,
    const stride_span_z &modulus,
    QubitOrTrue control) {
    size_t n = target.size();
    throw_unless(modulus.size() > 0, "modulus.size() == 0 (division by zero will occur because modulus must be zero)");
    throw_unless(modulus.size() == n, "modulus.size() != target.size()");
    throw_unless(modulus.front() == true, "modulus must be odd (modulus[0] == true)");
    throw_unless(modulus.back() == true, "modulus must be full capacity (modulus[target.size() - 1] == true)");

    auto mark = builder.raii_mark_block_entry(control.is_qubit() ? "c_imul2_inv_mod_inline" : "imul2_inv_mod_inline");

    // Trivial reductions.
    if (target.size() == 1) {
        // Only valid modulus is 1, so the value must be 0, so multiplication by 2 does nothing.
        return;
    } else if (target.size() == 2) {
        // Only valid modulus is 3, so multiplying by 2 just exchanges 01 and 10.
        builder.cswap(control, target[0], target[1]);
        return;
    }

    auto workspace = ctx.take_clean(n - 2);
    QubitId control_and_target0 = target[0];
    if (control.is_qubit()) {
        control_and_target0 = ctx.take_clean(1)[0];
    }

    if (control.is_qubit()) {
        builder.reset_and(control, target[0], control_and_target0);
    }
    builder.cx(control_and_target0, target[1]);
    builder.ccx(control_and_target0, modulus[1], target[1]);
    builder.x(target[0]);
    if (control.is_qubit()) {
        builder.cx(control, control_and_target0);
    }
    builder.ccx(modulus[1], target[1], workspace[0]);
    builder.cx(target[1], workspace[0]);
    builder.x(workspace[0]);

    for (size_t k = n - 1; k-- > 2;) {
        builder.ccx(control, modulus[k], target[k]);
    }

    // Reverse pass of controlled ripple carry subtraction, cancelling the subtraction on underflow.
    for (size_t k = 2; k < n - 1; k++) {
        builder.cx(modulus[k], workspace[k - 2]);
        builder.ccx(control_and_target0, workspace[k - 2], target[k]);
        builder.reset_and(target[k], workspace[k - 2], workspace[k - 1]);
        builder.cx(modulus[k], workspace[k - 1]);
    }

    if (control.is_qubit()) {
        builder.cx(control, control_and_target0);
    }
    builder.x(target[0]);
    builder.x(workspace[n - 3]);
    builder.ccx(control_and_target0, workspace[n - 3], target[n - 1]);
    if (control.is_qubit()) {
        builder.del_and(control, target[0], control_and_target0);
    }
    builder.cccx(control, target[n - 1], workspace[n - 3], target[0], ctx.clean_workspace);
    builder.x(workspace[n - 3]);

    for (size_t k = n - 1; k-- > 2;) {
        builder.cx(modulus[k], workspace[k - 1]);
        builder.del_and(target[k], workspace[k - 2], workspace[k - 1]);
        builder.cx(modulus[k], workspace[k - 2]);
        builder.ccx(control, workspace[k - 2], target[k]);
    }
    {
        auto mx = builder.hmr_raii_xbit(workspace[0]);
        builder.ccx(target[1], modulus[1], mx);
        builder.cx(target[1], mx);
        builder.x(mx);
    }

    builder.cright_rotate(control, target);
}

void kickmix::gen_imul2_inv_mod(
    CircuitBuilder &builder,
    CircuitGenCtx ctx,
    stride_span<const QubitId> target,
    const stride_span_z &modulus,
    QubitOrTrue control,
    double btol) {
    // Trivial reductions.
    if (target.size() == 1) {
        // Only valid modulus is 1, so the value must be 0, so multiplication by 2 does nothing.
        return;
    } else if (target.size() == 2) {
        // Only valid modulus is 3, so multiplying by 2 just exchanges 01 and 10.
        builder.cswap(control, target[0], target[1]);
        return;
    }
    size_t n = target.size();
    btol = std::max(btol, 0.0);
    if (btol >= n && ctx.clean_workspace.size() + 2 >= n && !ctx.minimize_qubits) {
        gen_imul2_inv_mod_with_subcmp_merged_by_inlining(builder, ctx, target, modulus, control);
        return;
    }
    auto mark = builder.raii_mark_block_entry(control.is_qubit() ? "c_imul2_inv_mod_approx" : "imul2_inv_mod_approx");
    std::vector<QubitOrBitOrBool> half_modulus;
    for (size_t k = 1; k < modulus.size(); k++) {
        half_modulus.push_back(modulus[k]);
    }

    auto ancs = ctx.take_clean(2, "gen_imul2_inv_mod");
    builder.reset(ancs[0]);
    builder.cx(target[0], ancs[0]);
    builder.reset_and(ancs[0], control, ancs[1]);
    builder.cright_rotate(control, target);

    builder.inplace_invert(half_modulus);
    while (half_modulus.back() == false) {
        half_modulus.pop_back();
    }

    size_t single_bit_comparison_btol = 0;
    while (single_bit_comparison_btol < n && modulus[n - single_bit_comparison_btol - 1] == true) {
        single_bit_comparison_btol += 1;
    }
    if (single_bit_comparison_btol <= btol + 20) {
        btol += 1;  // Two approximate subroutines.
    }

    gen_isub_classical(builder, ctx, target, half_modulus, false, ancs[1], btol);
    builder.inplace_invert(half_modulus);

    builder.del_and(ancs[0], control, ancs[1]);
    {
        auto mx = builder.hmr_raii_push_condition(ancs[0]);
        builder.cz(target[0], control);
        builder.z(target[0]);

        if (single_bit_comparison_btol > btol) {
            // When the modulus is slightly below a power of 2, the wraparound bit is an
            // excellent predictor of the comparison.
            builder.cz(control, target.back());
        } else {
            while (half_modulus.size() < target.size()) {
                half_modulus.push_back(true);
            }
            half_modulus.back() = false;
            gen_flip_if_gt(builder, ctx, target, half_modulus, MINUS_KET, false, control, btol);
        }
    }
}
