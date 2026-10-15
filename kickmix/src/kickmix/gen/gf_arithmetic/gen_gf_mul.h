#ifndef _KICKMIX_GEN_GF_MUL_H
#define _KICKMIX_GEN_GF_MUL_H

#include "kickmix/build/circuit_builder.h"
#include "kickmix/build/circuit_gen_ctx.h"
#include "kickmix/util/gf2_field.h"

namespace kickmix {

/// Adds the product of two GF(2^m) registers into a third one.
///
/// As pseudocode:
///     Q_target ^= Q_lhs * Q_rhs  (mod the field polynomial)
///
/// The product is computed with a recursive Karatsuba decomposition, costing O(m^1.585) Toffolis
/// instead of the m^2 Toffolis a schoolbook product would need. At m = 512 that is the difference
/// between roughly twenty thousand and a quarter million Toffolis.
///
/// The decomposition is entirely in place: the three half sized products are accumulated directly
/// into Q_target, and the "multiply by 1 + x^k" bookkeeping between them is done with the
/// CNOT-only routines gen_gf_imul_x and gen_gf_imul_classical. So an uncontrolled product needs no
/// workspace at all. A controlled product needs m clean qubits, because it computes the product
/// into a temporary register, adds that in under the control, and then uncomputes it.
///
/// Args:
///     builder: Where to append the circuit operations.
///     ctx: Preferences and resources for the operation to use.
///     field: The field the registers hold elements of.
///     Q_target: The register to add the product into. Its size must equal the field degree.
///     Q_lhs: The first factor. Left unchanged.
///     Q_rhs: The second factor. Left unchanged.
///     control: Determines if the operation happens or not.
///
/// Requires:
///     Q_target, Q_lhs, and Q_rhs all have size field.degree(), and are pairwise disjoint.
///     ctx.clean_workspace has at least field.degree() qubits, if the operation is controlled.
///     Register disjointness is a precondition, not something the generators verify. Scanning
///     the registers on every call would cost more than the gates being emitted, so passing
///     overlapping registers silently produces wrong arithmetic rather than an error.
void gen_gf_mul(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const QubitId> Q_target,
    stride_span<const QubitId> Q_lhs,
    stride_span<const QubitId> Q_rhs,
    QubitOrTrue control = true);

/// Clears a GF(2^m) register that is known to hold the product of two other registers.
///
/// As pseudocode:
///     assert Q_target == Q_lhs * Q_rhs
///     Q_target = 0
///
/// This is the same unitary as gen_gf_mul, but when the operation is uncontrolled it is realized by
/// measuring the target in the X basis and applying classically conditioned CZ fixups instead of by
/// running the product circuit backwards. That makes uncomputing a product completely free of
/// Toffolis, which is why the controlled path of gen_gf_mul can afford a temporary register.
///
/// Because the fixups are phase corrections that depend on measurement results, this routine is
/// only valid when the target really does hold the product. Feeding it anything else silently
/// applies an unintended phase.
///
/// Args:
///     builder: Where to append the circuit operations.
///     ctx: Preferences and resources for the operation to use.
///     field: The field the registers hold elements of.
///     Q_target: The register holding the product, which is returned to the zero state.
///     Q_lhs: The first factor. Left unchanged.
///     Q_rhs: The second factor. Left unchanged.
///     control: Determines if the operation happens or not. A controlled uncomputation falls back
///         to running gen_gf_mul, since the measurement trick needs the target to be known.
///
/// Requires:
///     Q_target, Q_lhs, and Q_rhs all have size field.degree(), and are pairwise disjoint.
///     Register disjointness is a precondition, not something the generators verify. Scanning
///     the registers on every call would cost more than the gates being emitted, so passing
///     overlapping registers silently produces wrong arithmetic rather than an error.
void gen_gf_unmul(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const QubitId> Q_target,
    stride_span<const QubitId> Q_lhs,
    stride_span<const QubitId> Q_rhs,
    QubitOrTrue control = true);

/// Negates the amplitudes of states where a masked parity of a GF(2^m) product is odd.
///
/// As pseudocode:
///     phase *= (-1) ** popcount(B_mask & (Q_lhs * Q_rhs))
///
/// This is the phase kickback primitive that gen_gf_unmul is built out of, exposed separately
/// because it is also what is needed to fold a product into a phase without ever materializing it.
///
/// The product bits are never computed. Instead the low half of the unreduced product contributes
/// directly (bit i of the low half is hit by mask bit i), while the high half contributes through
/// the transpose of the "multiply by x^m" matrix, whose columns are folded into one parity bit at a
/// time. Each column costs about two classical CNOTs per set entry, so the folding is cheap for the
/// low weight default moduli and grows toward O(m^2) classical CNOTs for dense custom moduli. The
/// quantum cost (m^2 classically conditioned CZs) does not depend on the modulus.
///
/// Args:
///     builder: Where to append the circuit operations.
///     ctx: Preferences and resources for the operation to use.
///     field: The field the registers hold elements of.
///     B_mask: Classical bits selecting which product bits contribute to the phase.
///     Q_lhs: The first factor. Left unchanged.
///     Q_rhs: The second factor. Left unchanged.
///
/// Requires:
///     B_mask, Q_lhs, and Q_rhs all have size field.degree(), and Q_lhs and Q_rhs are disjoint.
///     Register disjointness is a precondition, not something the generators verify. Scanning
///     the registers on every call would cost more than the gates being emitted, so passing
///     overlapping registers silently produces wrong arithmetic rather than an error.
void gen_gf_phase_by_product(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const BitId> B_mask,
    stride_span<const QubitId> Q_lhs,
    stride_span<const QubitId> Q_rhs);

/// Adds the carry-less product of two registers into a third one, without any field reduction.
///
/// As pseudocode:
///     Q_target ^= Q_lhs * Q_rhs  (as polynomials over GF(2))
///
/// Uses the in-place Karatsuba decomposition of Kepley and Steinwandt, which needs no ancillas: the
/// three half sized subproducts are accumulated into overlapping windows of the target, and the
/// overlaps are corrected by CNOT ladders that move sums of target bits around.
///
/// Args:
///     builder: Where to append the circuit operations.
///     ctx: Preferences and resources for the operation to use.
///     Q_target: The register to add the product into. Its size must be 2 * Q_lhs.size() - 1.
///     Q_lhs: The first factor. Left unchanged.
///     Q_rhs: The second factor. Left unchanged.
///
/// Requires:
///     Q_lhs.size() == Q_rhs.size(), and Q_target.size() == 2 * Q_lhs.size() - 1, and the three
///     registers are pairwise disjoint.
///     Register disjointness is a precondition, not something the generators verify. Scanning
///     the registers on every call would cost more than the gates being emitted, so passing
///     overlapping registers silently produces wrong arithmetic rather than an error.
void gen_gf2_poly_mul(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    stride_span<const QubitId> Q_target,
    stride_span<const QubitId> Q_lhs,
    stride_span<const QubitId> Q_rhs);

}  // namespace kickmix

#endif
