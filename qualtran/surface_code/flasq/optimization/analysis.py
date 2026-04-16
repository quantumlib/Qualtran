#  Copyright 2024 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import itertools
from functools import lru_cache
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import sympy
from frozendict import frozendict

from qualtran.resource_counting import get_cost_value, QubitCount
from qualtran.surface_code.flasq import cultivation_analysis
from qualtran.surface_code.flasq.cirq_interop import convert_circuit_for_flasq_analysis
from qualtran.surface_code.flasq.flasq_model import (
    apply_flasq_cost_model,
    FLASQCostModel,
    FLASQSummary,
    get_rotation_depth,
)
from qualtran.surface_code.flasq.measurement_depth import MeasurementDepth, TotalMeasurementDepth
from qualtran.surface_code.flasq.optimization.configs import CoreParametersConfig, ErrorBudget
from qualtran.surface_code.flasq.span_counting import GateSpan, TotalSpanCost
from qualtran.surface_code.flasq.symbols import MIXED_FALLBACK_T_COUNT, ROTATION_ERROR
from qualtran.surface_code.flasq.utils import substitute_until_fixed_point
from qualtran.surface_code.flasq.volume_counting import FLASQGateCounts, FLASQGateTotals


@lru_cache(maxsize=None)
def analyze_logical_circuit(
    *,
    circuit_builder_func: Callable,
    circuit_builder_kwargs: frozendict,
    total_allowable_rotation_error: float,
) -> Optional[frozendict]:
    """Builds the logical quantum circuit and analyzes its abstract properties.

    This function is cached based on its arguments. It calls the provided
    `circuit_builder_func` and handles two possible return types:

    1.  A `cirq.Circuit` object (or `None`).
    2.  A tuple of `(cirq.Circuit, Dict[str, Any])` (or `None`). The dictionary
        contains keyword arguments that will be passed to
        `convert_circuit_for_flasq_analysis`. This is useful for circuits
        that require special handling during decomposition, like those with
        custom Bloqs that need a `qubit_manager`.

    Args:
        circuit_builder_func: The function that constructs the cirq.Circuit.
        circuit_builder_kwargs: Hashable (frozendict) keyword arguments for the builder.
        total_allowable_rotation_error: Total error budget for all rotations in the circuit.

    Returns:
        A frozendict containing the analysis results, or None if the circuit
        builder returns None. The dictionary includes:
            - flasq_counts: FLASQGateCounts object.
            - total_span: GateSpan object.
            - measurement_depth: MeasurementDepth object.
            - individual_allowable_rotation_error: Calculated error per rotation.
            - qubit_counts: The number of algorithmic qubits used by the circuit.
    """
    builder_return_val = circuit_builder_func(**circuit_builder_kwargs)

    if builder_return_val is None:
        return None

    if isinstance(builder_return_val, tuple):
        original_circuit, conversion_kwargs = builder_return_val
    else:
        original_circuit, conversion_kwargs = builder_return_val, {}

    cbloq, decomposed_circuit = convert_circuit_for_flasq_analysis(
        original_circuit, **conversion_kwargs
    )

    flasq_counts: FLASQGateCounts = get_cost_value(cbloq, FLASQGateTotals())
    total_span: GateSpan = get_cost_value(cbloq, TotalSpanCost())
    qubit_counts = get_cost_value(cbloq, QubitCount())

    if total_allowable_rotation_error == 0:
        if flasq_counts.total_rotations != 0:
            raise ValueError(
                "total_allowable_rotation_error cannot be 0 if there are rotations in the circuit, "
                "as this implies infinitely precise rotations."
            )

    if flasq_counts.total_rotations == 0:
        ind_rot_err = 1.0
        rotation_depth_val: Union[float, sympy.Expr] = 0.0
    else:
        ind_rot_err = total_allowable_rotation_error / flasq_counts.total_rotations
        rotation_depth_val = get_rotation_depth(rotation_error=ind_rot_err)

    measurement_depth_obj: MeasurementDepth = get_cost_value(
        cbloq, TotalMeasurementDepth(rotation_depth=rotation_depth_val)
    )

    return frozendict(
        {
            "flasq_counts": flasq_counts,
            "total_span": total_span,
            "measurement_depth": measurement_depth_obj,
            "individual_allowable_rotation_error": ind_rot_err,
            "qubit_counts": qubit_counts,
            "flasq_conversion_kwargs": frozendict(conversion_kwargs),
        }
    )


@lru_cache(maxsize=None)
def calculate_single_flasq_summary(
    *,
    logical_circuit_analysis: frozendict,
    n_phys_qubits: int,
    code_distance: int,
    flasq_model_obj: FLASQCostModel,
    logical_timesteps_per_measurement: float,
) -> Optional[FLASQSummary]:
    """Applies a FLASQ cost model to logical circuit properties.

    Args:
        logical_circuit_analysis: Cached analysis results from
            ``analyze_logical_circuit``.
        n_phys_qubits: Total physical qubit count. Historical choice;
            n_phys_qubits and n_logical_qubits are derivable from each other
            given code_distance: n_phys_per_logical = 2*(d+1)^2.
        code_distance: Surface code distance d.
        flasq_model_obj: The ``FLASQCostModel`` to apply.
        logical_timesteps_per_measurement: Scales raw measurement depth by
            reaction time. Computed as reaction_time_in_cycles / code_distance.
    """
    flasq_counts = logical_circuit_analysis["flasq_counts"]
    total_span = logical_circuit_analysis["total_span"]
    measurement_depth_obj = logical_circuit_analysis["measurement_depth"]
    qubit_counts = logical_circuit_analysis["qubit_counts"]

    n_total_logical_qubits = n_phys_qubits // (2 * (code_distance + 1) ** 2)

    if n_total_logical_qubits - qubit_counts <= 0:
        return None

    flasq_summary = apply_flasq_cost_model(
        model=flasq_model_obj,
        n_total_logical_qubits=n_total_logical_qubits,
        qubit_counts=qubit_counts,
        counts=flasq_counts,
        span_info=total_span,
        measurement_depth=measurement_depth_obj,
        logical_timesteps_per_measurement=logical_timesteps_per_measurement,
    )
    return flasq_summary


def generate_circuit_specific_configs(
    circuit_builder: Callable,
    circuit_builder_kwargs: frozendict,
    total_synthesis_error: float,
    total_cultivation_error: float,
    phys_error_rate: float,
    reference_code_distance: int,
) -> Tuple[CoreParametersConfig, float]:
    """
    Derives circuit-specific core parameters for error budgeting.

    This function centralizes the logic for deriving rotation synthesis and
    T-gate cultivation parameters based on a specific circuit's structure and
    high-level error budgets.

    Args:
        circuit_builder: A callable that builds the target `cirq.Circuit`.
        circuit_builder_kwargs: Arguments for the `circuit_builder`.
        total_synthesis_error: The total allowable error from rotation synthesis.
        total_cultivation_error: The total allowable error from T-gate cultivation.
        phys_error_rate: The physical error rate assumption.
        reference_code_distance: A reference code distance for deriving `vcult_factor`.

    Returns:
        A tuple containing:
            - The derived `CoreParametersConfig`.
            - The `total_synthesis_error` that was used (for convenience).
    """
    logical_analysis = analyze_logical_circuit(
        circuit_builder_func=circuit_builder,
        circuit_builder_kwargs=circuit_builder_kwargs,
        total_allowable_rotation_error=total_synthesis_error,
    )
    individual_allowable_rotation_error = logical_analysis["individual_allowable_rotation_error"]
    summary = apply_flasq_cost_model(
        model=FLASQCostModel(),
        n_total_logical_qubits=logical_analysis["qubit_counts"],
        qubit_counts=logical_analysis["qubit_counts"],
        counts=logical_analysis["flasq_counts"],
        span_info=logical_analysis["total_span"],
        measurement_depth=logical_analysis["measurement_depth"],
        logical_timesteps_per_measurement=0,
    )
    total_t_count = substitute_until_fixed_point(
        summary.total_t_count,
        frozendict({ROTATION_ERROR: individual_allowable_rotation_error}),
        try_make_number=True,
    )
    target_individual_t_gate_error = total_cultivation_error / total_t_count
    core_config = CoreParametersConfig.from_cultivation_analysis(
        physical_error_rate=phys_error_rate,
        target_individual_t_gate_error=target_individual_t_gate_error,
        reference_code_distance=reference_code_distance,
    )
    return (core_config, total_synthesis_error)


def generate_configs_for_constrained_qec(
    circuit_builder_func: Callable,
    circuit_builder_kwargs: frozendict,
    error_budget: ErrorBudget,
    phys_error_rate_list: Iterable[float],
    code_distance_list: Iterable[int],
    cultivation_data_decimal_precision: int = 8,
    cultivation_data_uncertainty_cutoff: Optional[float] = 100,
    round_error_rate_up_to_simulated_cultivation_data: bool = True,
) -> List[CoreParametersConfig]:
    """
    Generates CoreParametersConfig optimized for the constrained QEC approach.

    Determines the required p_mag based on T-count and epsilon_cult, finds the
    optimal cultivation strategy, and generates the sweep space over d and p_phys.
    """
    logical_analysis = analyze_logical_circuit(
        circuit_builder_func=circuit_builder_func,
        circuit_builder_kwargs=circuit_builder_kwargs,
        total_allowable_rotation_error=error_budget.synthesis,
    )

    if logical_analysis is None:
        return []

    flasq_counts = logical_analysis["flasq_counts"]
    ind_rot_err = logical_analysis["individual_allowable_rotation_error"]

    M_expr = (
        flasq_counts.t
        + flasq_counts.toffoli * 4
        + flasq_counts.and_gate * 4
        + MIXED_FALLBACK_T_COUNT * (flasq_counts.z_rotation + flasq_counts.x_rotation)
    )

    M = float(
        substitute_until_fixed_point(
            M_expr, frozendict({ROTATION_ERROR: ind_rot_err}), try_make_number=True
        )
    )

    if M <= 0:
        required_p_mag = 1.0
    else:
        required_p_mag = error_budget.cultivation / M

    configs = []

    for p_phys in phys_error_rate_list:
        phys_err_for_cult_lookup = p_phys
        if round_error_rate_up_to_simulated_cultivation_data:
            rounded_err = cultivation_analysis.round_error_rate_up(
                p_phys, cultivation_data_decimal_precision
            )
            if rounded_err is None:
                continue
            phys_err_for_cult_lookup = rounded_err

        best_cult_params = cultivation_analysis.find_best_cultivation_parameters(
            physical_error_rate=phys_err_for_cult_lookup,
            target_logical_error_rate=required_p_mag,
            decimal_precision=cultivation_data_decimal_precision,
            uncertainty_cutoff=cultivation_data_uncertainty_cutoff,
        )

        if best_cult_params.empty:
            continue

        cult_error_rate = best_cult_params["t_gate_cultivation_error_rate"]
        expected_volume = best_cult_params["expected_volume"]
        source_distance = int(best_cult_params["cultivation_distance"])

        for d in code_distance_list:
            vcult_factor = expected_volume / (2 * (d + 1) ** 2 * d)

            configs.append(
                CoreParametersConfig(
                    code_distance=d,
                    phys_error_rate=p_phys,
                    cultivation_error_rate=cult_error_rate,
                    vcult_factor=vcult_factor,
                    cultivation_data_source_distance=source_distance,
                )
            )
    return configs
