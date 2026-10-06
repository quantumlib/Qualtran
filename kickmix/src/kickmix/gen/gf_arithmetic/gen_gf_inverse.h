#ifndef _KICKMIX_GEN_GF_INVERSE_H
#define _KICKMIX_GEN_GF_INVERSE_H

#include "kickmix/build/circuit_builder.h"
#include "kickmix/build/circuit_gen_ctx.h"
#include "kickmix/util/gf2_field.h"

namespace kickmix {

/// Returns how many clean workspace qubits self-cleaning gen_gf_inverse and gen_gf_uninverse need
/// from ctx.clean_workspace when Q_chain is not passed explicitly.
///
/// The Itoh-Tsujii chain keeps one register per addition chain step alive at once, so the workspace
/// grows like degree * log2(degree). For GF(2^512) that is sixteen extra registers.
size_t gf_inverse_workspace_size(const GF2Field &field);

/// Returns the number of qubits needed in the explicit Q_chain register when using the split
/// gen_gf_inverse(..., Q_chain) and gen_gf_uninverse(..., Q_chain) overloads.
size_t gf_inverse_chain_size(const GF2Field &field);

/// Computes the multiplicative inverse of a GF(2^m) register into a zeroed register, cleaning up
/// all intermediate chain registers before returning.
///
/// As pseudocode:
///     assert Q_target == 0
///     Q_target = Q_input ** -1  (mod the field polynomial, with 0 mapping to 0)
///
/// Uses the Itoh-Tsujii algorithm. Since the multiplicative group has order 2^m - 1, the inverse is
/// the power 2^m - 2 = 2 * (2^(m-1) - 1), so the whole job is computing
///     beta(m - 1) = Q_input ** (2^(m-1) - 1)
/// and squaring it once. The beta values satisfy
///     beta(a + b) = beta(a)^(2^b) * beta(b)
/// so beta(m - 1) can be reached by an addition chain for m - 1. The chain used here is the binary
/// one: repeatedly double to get beta(2^i) for every i up to log2(m - 1), then fold in one term per
/// set bit of m - 1. That costs only about log2(m) + popcount(m - 1) field multiplications, versus
/// the 2m multiplications of naive exponentiation.
///
/// Raising to a power of two is free of Toffolis (it is the GF(2)-linear Frobenius map), which is
/// what makes this chain so much cheaper than a generic addition chain would be.
///
/// Zero has no inverse; it is mapped to zero, which keeps the operation reversible.
///
/// Args:
///     builder: Where to append the circuit operations.
///     ctx: Preferences and resources for the operation to use.
///     field: The field the registers hold elements of.
///     Q_target: The register receiving the inverse. Must start in the zero state.
///     Q_input: The register to invert. Left unchanged.
///
/// Requires:
///     Q_target.size() == Q_input.size() == field.degree(), and they are disjoint.
///     ctx.clean_workspace has at least gf_inverse_workspace_size(field) qubits.
///     Register disjointness is a precondition, not something the generators verify. Scanning
///     the registers on every call would cost more than the gates being emitted, so passing
///     overlapping registers silently produces wrong arithmetic rather than an error.
void gen_gf_inverse(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const QubitId> Q_target,
    stride_span<const QubitId> Q_input);

/// Computes the multiplicative inverse of a GF(2^m) register into Q_target, storing the addition
/// chain's intermediate values in the caller-supplied Q_chain register so that a subsequent call to
/// gen_gf_uninverse(..., Q_chain) can unwind the inversion with zero additional Toffolis.
///
/// Requires:
///     Q_target.size() == Q_input.size() == field.degree().
///     Q_chain.size() == gf_inverse_chain_size(field).
///     Q_target and Q_chain start in the zero state, and all three registers are disjoint.
///     Register disjointness is a precondition, not something the generators verify. Scanning
///     the registers on every call would cost more than the gates being emitted, so passing
///     overlapping registers silently produces wrong arithmetic rather than an error.
void gen_gf_inverse(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const QubitId> Q_target,
    stride_span<const QubitId> Q_input,
    stride_span<const QubitId> Q_chain);

/// Clears a register known to hold the inverse of another register, rebuilding and uncomputing the
/// addition chain using clean workspace from ctx.
///
/// As pseudocode:
///     assert Q_target == Q_input ** -1
///     Q_target = 0
void gen_gf_uninverse(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const QubitId> Q_target,
    stride_span<const QubitId> Q_input);

/// Clears Q_target (holding Q_input ** -1) and Q_chain (holding the live addition chain
/// intermediates produced by gen_gf_inverse(..., Q_chain)) back to the zero state.
///
/// Because Q_chain already holds the intermediate chain values, this uncomputation requires zero
/// Toffoli gates (every chain step is undone via measurement-based gen_gf_unmul).
void gen_gf_uninverse(
    CircuitBuilder &builder,
    const CircuitGenCtx &ctx,
    const GF2Field &field,
    stride_span<const QubitId> Q_target,
    stride_span<const QubitId> Q_input,
    stride_span<const QubitId> Q_chain);

}  // namespace kickmix

#endif
