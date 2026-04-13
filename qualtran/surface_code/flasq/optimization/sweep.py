import itertools
from typing import Callable, Iterable, List, Tuple, Union

import attrs
import numpy as np
from frozendict import frozendict
from qualtran.surface_code.flasq.flasq_model import FLASQCostModel, FLASQSummary
from qualtran.surface_code.flasq.optimization.analysis import (
    analyze_logical_circuit,
    calculate_single_flasq_summary,
)
from qualtran.surface_code.flasq.optimization.configs import CoreParametersConfig, _ensure_iterable
from qualtran.surface_code.flasq.symbols import ROTATION_ERROR, T_REACT, V_CULT_FACTOR
from tqdm.auto import tqdm


@attrs.frozen
class SweepResult:
    """Raw, symbolic results of a single FLASQ sweep point.

    Attributes:
        circuit_builder_kwargs: Arguments used to build the circuit.
        core_config: The ``CoreParametersConfig`` for this point.
        total_allowable_rotation_error: Total rotation synthesis error budget.
        reaction_time_in_cycles: Reaction time expressed in surface code cycles.
        flasq_model_config: ``(FLASQCostModel, name)`` tuple.
        n_phys_qubits: Total physical qubit count. Historical choice;
            n_phys_qubits and n_logical_qubits are derivable from each other
            given code_distance: n_phys_per_logical = 2*(d+1)^2.
        logical_circuit_analysis: Cached analysis of the logical circuit.
        flasq_summary: Symbolic ``FLASQSummary`` before assumption substitution.
    """

    circuit_builder_kwargs: frozendict
    core_config: CoreParametersConfig
    total_allowable_rotation_error: float
    reaction_time_in_cycles: float
    flasq_model_config: Tuple[FLASQCostModel, str]
    n_phys_qubits: int
    logical_circuit_analysis: frozendict
    flasq_summary: FLASQSummary

    def get_assumptions(self) -> frozendict:
        """
        Consolidates the logic for creating the assumptions dictionary required
        to resolve symbols in the flasq_summary.
        """
        flasq_model_obj, _ = self.flasq_model_config
        t_react_val = self.reaction_time_in_cycles / self.core_config.code_distance

        assumptions = frozendict(
            {
                ROTATION_ERROR: self.logical_circuit_analysis[
                    "individual_allowable_rotation_error"
                ],
                V_CULT_FACTOR: self.core_config.vcult_factor,
                T_REACT: t_react_val,
            }
        )
        return assumptions


def run_sweep(
    *,
    circuit_builder_func: Callable,
    circuit_builder_kwargs_list: Union[frozendict, Iterable[frozendict]],
    core_configs_list: Union[CoreParametersConfig, Iterable[CoreParametersConfig]],
    total_allowable_rotation_error_list: Union[float, Iterable[float]],
    reaction_time_in_cycles_list: Union[float, Iterable[float]],
    flasq_model_configs: Iterable[Tuple[FLASQCostModel, str]],
    n_phys_qubits_total_list: Union[int, Iterable[int]],
    print_level: int = 1,
) -> List[SweepResult]:
    """
    Core sweep function to generate physical cost estimates (`FLASQSummary`) over a parameter space.

    This function is the primary "engine" for parameter sweeps. It iterates through all
    combinations of the provided parameters, calculates the logical resource requirements,
    and then applies the FLASQ cost model to produce a `FLASQSummary` for each valid point.

    TODO: The current API expects a physical qubit count (n_phys_qubits) and
    derives the logical qubit budget. Historical choice; n_phys_qubits and
    n_logical_qubits are derivable from each other given code_distance:
    n_phys_per_logical = 2*(d+1)^2. A future refactor should allow users to
    provide ``n_logical_qubits_total_list`` directly.

    The output of this function is a list of `SweepResult` objects, each containing
    all the input parameters for a given sweep point, along with the resulting symbolic
    `FLASQSummary`. This data can then be used by post-processing functions.

    Args:
        circuit_builder_func: Function to build the `cirq.Circuit`.
        circuit_builder_kwargs_list: A single `frozendict` or an iterable of `frozendict`
            objects, each representing a set of keyword arguments for `circuit_builder_func`.
        core_configs_list: A single `CoreParametersConfig` or an iterable of them, defining
            interdependent parameters like code distance and error rates.
        total_allowable_rotation_error_list: Single value or iterable of total rotation errors.
        reaction_time_in_cycles_list: Single value or iterable for reaction time in cycles.
        flasq_model_configs: Iterable of (FLASQCostModel, str_name) tuples.
        n_phys_qubits_total_list: Single value or iterable of total physical qubit counts.
        print_level: Controls verbosity (0: silent, 1: progress bar, 2: details).

    Returns:
        A list of `SweepResult` objects, each containing the parameters for a sweep point
        and the resulting symbolic `FLASQSummary` object.
    """
    sweep_results = []

    param_iterables = [
        _ensure_iterable(
            circuit_builder_kwargs_list, treat_frozendict_as_single_item=True
        ),
        _ensure_iterable(core_configs_list),
        _ensure_iterable(total_allowable_rotation_error_list),
        _ensure_iterable(reaction_time_in_cycles_list),
        _ensure_iterable(flasq_model_configs),
        _ensure_iterable(n_phys_qubits_total_list),
    ]

    param_lists = [list(it) for it in param_iterables]
    total_iterations = np.prod([len(p_list) for p_list in param_lists])

    iterable_product = itertools.product(*param_lists)
    progress_bar = tqdm(
        iterable_product,
        total=total_iterations,
        desc="FLASQ Sweep",
        leave=True,
        disable=(print_level < 1),
    )

    for params_combo in progress_bar:
        (
            circuit_kwargs,
            core_config,
            total_rot_error,
            reaction_time,
            flasq_model_config,
            n_phys_qubits,
        ) = params_combo

        logical_analysis = analyze_logical_circuit(
            circuit_builder_func=circuit_builder_func,
            circuit_builder_kwargs=circuit_kwargs,
            total_allowable_rotation_error=total_rot_error,
        )

        if logical_analysis is None:
            if print_level >= 2:
                progress_bar.write(f"Skipping non-viable circuit: {params_combo}")
            continue

        flasq_model_obj, _ = flasq_model_config
        logical_timesteps_per_measurement = (
            reaction_time / core_config.code_distance
        )

        summary = calculate_single_flasq_summary(
            logical_circuit_analysis=logical_analysis,
            n_phys_qubits=n_phys_qubits,
            code_distance=core_config.code_distance,
            flasq_model_obj=flasq_model_obj,
            logical_timesteps_per_measurement=logical_timesteps_per_measurement,
        )

        if summary is None:
            if print_level >= 2:
                progress_bar.write(f"Skipping non-viable configuration: {params_combo}")
            continue

        sweep_results.append(
            SweepResult(
                circuit_builder_kwargs=circuit_kwargs,
                core_config=core_config,
                total_allowable_rotation_error=total_rot_error,
                reaction_time_in_cycles=reaction_time,
                flasq_model_config=flasq_model_config,
                n_phys_qubits=n_phys_qubits,
                logical_circuit_analysis=logical_analysis,
                flasq_summary=summary,
            )
        )

    return sweep_results
