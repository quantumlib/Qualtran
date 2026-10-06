#ifndef _KICKMIX_GEN_GF_DIV_H
#define _KICKMIX_GEN_GF_DIV_H

#include "kickmix/build/circuit_builder.h"
#include "kickmix/build/circuit_gen_ctx.h"
#include "kickmix/util/gf2_field.h"

namespace kickmix {

/// Returns how many clean workspace qubits gen_gf_div and gen_gf_undiv need.
size_t gf_div_workspace_size(const GF2Field &field);

/// Adds the quotient of two GF(2^m) registers into a third one.
///
/// As pseudocode:
///     Q_target ^= Q_lhs / Q_rhs  (mod the field polynomial, with a zero divisor giving zero)
///
/// Division is inversion followed by multiplication. The inverse and its Itoh-Tsujii chain
/// intermediates are computed into borrowed clean workspace (Q_inverse and Q_chain) and then
/// unwound with gen_gf_uninverse(..., Q_chain), which skips the chain cleanup and rebuild that a
/// pair of self-cleaning inversions would otherwise perform.
///
/// Args:
///     builder: Where to append the circuit operations.
///     ctx: Preferences and resources for the operation to use.
///     field: The field the registers hold elements of.
///     Q_target: The register to add the quotient into. Its size must equal the field degree.
///     Q_lhs: The dividend. Left unchanged.
///     Q_rhs: The divisor. Left unchanged.
///
/// Requires:
///     All three registers have size field.degree(). Q_target must be disjoint from Q_lhs and
///     Q_rhs; Q_lhs and Q_rhs may be the same register, since the divisor is inverted into
///     borrowed workspace rather than in place.
///     ctx.clean_workspace has at least gf_div_workspace_size(field) qubits.
///     Register disjointness is a precondition, not something the generators verify. Scanning
///     the registers on every call would cost more than the gates being emitted, so passing
///     overlapping registers silently produces wrong arithmetic rather than an error.
void gen_gf_div(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const QubitId> Q_target,
    stride_span<const QubitId> Q_lhs,
    stride_span<const QubitId> Q_rhs);

void gen_gf_div(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const QubitId> Q_target,
    stride_span<const QubitId> Q_lhs,
    stride_span<const QubitId> Q_rhs,
    stride_span<const QubitId> Q_ancillas);

/// Clears a register known to hold the quotient of two other registers.
///
/// As pseudocode:
///     assert Q_target == Q_lhs / Q_rhs
///     Q_target = 0
///
/// This is the exact inverse of gen_gf_div, and is cheaper than it because the multiplication is
/// undone by measurement instead of by Toffolis.
void gen_gf_undiv(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const QubitId> Q_target,
    stride_span<const QubitId> Q_lhs,
    stride_span<const QubitId> Q_rhs);

void gen_gf_undiv(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const QubitId> Q_target,
    stride_span<const QubitId> Q_lhs,
    stride_span<const QubitId> Q_rhs,
    stride_span<const QubitId> Q_ancillas);

}  // namespace kickmix

#endif
