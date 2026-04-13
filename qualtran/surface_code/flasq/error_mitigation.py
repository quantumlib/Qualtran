"""PEC (probabilistic error cancellation) overhead estimation.

Computes the sampling overhead Gamma-squared, failure probabilities, and
wall-clock time per effective noiseless sample for a given FLASQ summary
and physical error parameters.
"""

from typing import Tuple
from functools import lru_cache

import sympy

from qualtran.symbolics import SymbolicFloat

from qualtran_flasq.flasq_model import FLASQSummary

ERROR_PER_CYCLE_PREFACTOR = 0.03  # c_cyc in the paper. Empirical prefactor for surface code logical error rate.


@lru_cache(maxsize=None)
def calculate_error_mitigation_metrics(
    flasq_summary: FLASQSummary,
    time_per_surface_code_cycle: SymbolicFloat,
    code_distance: SymbolicFloat,  # Can be int or float, using SymbolicFloat for flexibility
    lambda_val: SymbolicFloat,  # Renamed to avoid conflict with Python's lambda keyword
    cultivation_error_rate: SymbolicFloat,
) -> Tuple[SymbolicFloat, SymbolicFloat, SymbolicFloat]:
    """Calculates PEC overhead and wall-clock time per noiseless sample.

    Args:
        flasq_summary: The FLASQSummary object containing resource estimates.
        time_per_surface_code_cycle: The time taken for one cycle of the surface code.
        code_distance: The distance d of the surface code.
        lambda_val: Λ = p_th / p_phys in the paper, where p_th ≈ 0.01
            is the surface code threshold.
        cultivation_error_rate: p_mag in the paper. Error rate per magic
            state from cultivation.

    Returns:
        A tuple of (effective_time_per_noiseless_sample,
        wall_clock_time_per_sample, gamma_per_block).
        gamma_per_block: per-logical-timestep sampling penalty factor,
            computed as gamma_per_cycle ** code_distance. Note: the full
            circuit Γ² is computed internally but not returned.
    """
    # error_per_surface_code_cycle is the physical error rate per qubit per surface code cycle
    error_per_surface_code_cycle = ERROR_PER_CYCLE_PREFACTOR * lambda_val ** (
        -(code_distance + 1) / 2
    )

    # Gamma factors represent (1 - error_prob)^-1, effectively an error penalty.
    # For individual qubit errors per cycle
    # ERROR_PER_CYCLE in the formula corresponds to error_per_surface_code_cycle
    gamma_z_per_qubit_cycle = (1 - error_per_surface_code_cycle) ** -1
    gamma_x_per_qubit_cycle = (
        1 - error_per_surface_code_cycle
    ) ** -1  # Assuming symmetric X/Z errors

    # Combined gamma factor per surface code cycle for a logical qubit
    gamma_per_cycle = gamma_z_per_qubit_cycle * gamma_x_per_qubit_cycle

    # Gamma factor for T gates
    # ERROR_PER_T_GATE in the formula corresponds to cultivation_error_rate
    error_per_t_gate = cultivation_error_rate
    gamma_per_t_gate = (
        1 - 2 * error_per_t_gate
    ) ** -1  # The factor of 2 might be model-specific

    # Gamma factor per logical block (logical timestep)
    # CYCLES_PER_LOGICAL_TIMESTEP in the formula corresponds to code_distance

    total_regular_spacetime_cycles = (
        flasq_summary.regular_spacetime_volume
    ) * code_distance

    log_gamma_per_circuit = (
        sympy.log(gamma_per_cycle) * total_regular_spacetime_cycles
        + sympy.log(gamma_per_t_gate) * flasq_summary.total_t_count
    )

    gamma_per_circuit = sympy.exp(log_gamma_per_circuit)
    sampling_overhead = gamma_per_circuit**2

    wall_clock_time_per_sample = (
        time_per_surface_code_cycle * code_distance * flasq_summary.total_depth
    )
    effective_time_per_noiseless_sample = wall_clock_time_per_sample * sampling_overhead

    # TODO: Add the actual logical error rate as a metric to return in the future.

    return (
        effective_time_per_noiseless_sample,
        wall_clock_time_per_sample,
        gamma_per_cycle**code_distance,
    )


@lru_cache(maxsize=None)
def calculate_failure_probabilities(
    flasq_summary: FLASQSummary,
    code_distance: SymbolicFloat,
    lambda_val: SymbolicFloat,
    cultivation_error_rate: SymbolicFloat,
    error_prefactor: SymbolicFloat = ERROR_PER_CYCLE_PREFACTOR,
) -> Tuple[SymbolicFloat, SymbolicFloat]:
    """
    Calculates failure probabilities (P_fail) for Clifford operations and T-gates.

    Computes the exact probability of at least one failure using the binomial
    complement formula: P_fail = 1 - (1 - p)^N, where p is the per-event error
    rate and N is the number of independent error sites (spacetime cycles for
    Cliffords, T-gate count for T-gates).

    Args:
        flasq_summary: Resolved FLASQSummary object.
        code_distance: The distance d (also used as r, cycles per timestep).
        lambda_val: The lambda factor Λ.
        cultivation_error_rate: The error rate p_mag.
        error_prefactor: The prefactor c_cyc for the surface code error model.

    Returns:
        A tuple of (P_fail_Clifford, P_fail_T).
    """
    # 1. Calculate logical error rate per cycle (p_cyc)
    # p_cyc = c_cyc * Λ^(-(d+1)/2)
    p_cyc = error_prefactor * lambda_val ** (-(code_distance + 1) / 2)  # type: ignore

    # 2. Calculate P_fail_Clifford (Probability of at least one Clifford failure)
    # The relevant volume is the total spacetime volume except for that used by cultivation
    # (lattice surgery computation + idling) as this entire volume is exposed to
    # memory/logical errors and the cultivation errors are handled separately.
    # Total cycles = r * affected volume (where r=code_distance)

    total_regular_spacetime_cycles = (
        flasq_summary.regular_spacetime_volume
    ) * code_distance
    P_fail_Clifford = 1 - (1 - p_cyc) ** total_regular_spacetime_cycles

    # 3. Calculate P_fail_T (Probability of at least one T-gate failure)
    # P_fail_T = 1 - (1 - p_mag)^M
    P_fail_T = 1 - (1 - cultivation_error_rate) ** flasq_summary.total_t_count
    return P_fail_Clifford, P_fail_T
