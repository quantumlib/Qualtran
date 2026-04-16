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

from typing import Dict, List
import pandas as pd
import numpy as np
import sympy
from frozendict import frozendict
from tqdm.auto import tqdm
from joblib import Parallel, delayed

from qualtran.surface_code.flasq.optimization.sweep import SweepResult
from qualtran.surface_code.flasq.optimization.configs import ErrorBudget
from qualtran.surface_code.flasq.utils import substitute_until_fixed_point, convert_sympy_exprs_in_df
from qualtran.surface_code.flasq.error_mitigation import (
    calculate_error_mitigation_metrics,
    calculate_failure_probabilities,
    ERROR_PER_CYCLE_PREFACTOR,
)
from qualtran.surface_code.flasq.symbols import ROTATION_ERROR, V_CULT_FACTOR, T_REACT


def _process_single_result_for_logical_depth(r: SweepResult) -> dict:
    resolved_summary = r.flasq_summary.resolve_symbols(r.get_assumptions())
    cost_model = r.flasq_model_config[0]
    t_cultivation_volume = substitute_until_fixed_point(
        cost_model.t_cultivation_volume,
        frozendict({V_CULT_FACTOR: r.core_config.vcult_factor}),
    )

    data = {
        "Depth (Logical Timesteps)": resolved_summary.total_depth,
        "Number of Logical Qubits": r.n_phys_qubits
        // (2 * (r.core_config.code_distance + 1) ** 2),
        "Total Computational Volume": resolved_summary.total_computational_volume,
        "T-Count": resolved_summary.total_t_count,
        "Total Rotation Count": resolved_summary.total_rotation_count,
        "Number of Algorithmic Qubits": resolved_summary.n_algorithmic_qubits,
        "Number of Fluid Ancilla Qubits": resolved_summary.n_fluid_ancilla,
        "Total Spacetime Volume": resolved_summary.total_spacetime_volume,
        "Total Allowable Rotation Error": r.total_allowable_rotation_error,
        "FLASQ Model": r.flasq_model_config[1],
        "Number of Physical Qubits": r.n_phys_qubits,
        "Individual Rotation Error": r.logical_circuit_analysis[
            "individual_allowable_rotation_error"
        ],
        "Target Individual T-gate Error": r.core_config.target_t_error,
        "V_cult_factor": r.core_config.vcult_factor,
        "Raw Measurement Depth": resolved_summary.measurement_depth_val,
        "Scaled Measurement Depth": resolved_summary.scaled_measurement_depth,
        "T Cultivation Volume": t_cultivation_volume,
    }
    problem_specific_params = {
        f"circuit_arg_{k}": v for k, v in r.circuit_builder_kwargs.items()
    }
    data.update(problem_specific_params)
    return data


def post_process_for_logical_depth(
    sweep_results: List[SweepResult],
    *,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Post-processes `run_sweep` output to extract logical resource costs.

    This function is designed for comparing the logical costs (like total depth
    and T-count) of different circuit implementations, independent of physical
    error correction parameters. It takes a list of `SweepResult` objects,
    resolves the symbolic expressions in each `FLASQSummary`, and returns a
    pandas DataFrame of the logical metrics.

    Args:
        sweep_results: The list of `SweepResult` objects returned by `run_sweep`.
        n_jobs: Number of parallel jobs for ``joblib.Parallel``. Defaults to
            ``-1`` (use all available CPUs). Set to ``1`` to disable
            parallelism.

    Returns:
        A pandas DataFrame containing the logical resource costs for each sweep point.
    """
    parallel_gen = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(_process_single_result_for_logical_depth)(r)
        for r in sweep_results
    )
    processed_results = list(tqdm(parallel_gen, total=len(sweep_results), desc="Post-processing Logical Depth"))
    return pd.DataFrame(processed_results)


def _process_single_result_for_pec(r: SweepResult, time_per_surface_code_cycle: float) -> dict:
    flasq_model_obj, flasq_model_name = r.flasq_model_config
    t_react_val = r.reaction_time_in_cycles / r.core_config.code_distance

    assumptions = frozendict(
        {
            ROTATION_ERROR: r.logical_circuit_analysis[
                "individual_allowable_rotation_error"
            ],
            V_CULT_FACTOR: r.core_config.vcult_factor,
            T_REACT: t_react_val,
        }
    )
    resolved_summary = r.flasq_summary.resolve_symbols(assumptions)

    lambda_val = 1e-2 / r.core_config.phys_error_rate
    eff_time, wall_time, _ = calculate_error_mitigation_metrics(
        flasq_summary=resolved_summary,
        time_per_surface_code_cycle=time_per_surface_code_cycle,
        code_distance=r.core_config.code_distance,
        lambda_val=lambda_val,
        cultivation_error_rate=r.core_config.cultivation_error_rate,
    )

    problem_specific_params = frozendict(
        {f"circuit_arg_{k}": v for k, v in r.circuit_builder_kwargs.items()}
        | (
            {
                "circuit_arg_cultivation_data_source_distance": r.core_config.cultivation_data_source_distance,
            }
            if r.core_config.cultivation_data_source_distance is not None
            else {}
        )
    )
    data = {
        "Number of Physical Qubits": r.n_phys_qubits,
        "Physical Error Rate": r.core_config.phys_error_rate,
        "Code Distance": r.core_config.code_distance,
        "Number of Fluid Ancilla Qubits": resolved_summary.n_fluid_ancilla,
        "FLASQ Model": flasq_model_name.capitalize(),
        "Effective Time per Sample (s)": eff_time,
        "Wall Clock Time per Sample (s)": wall_time,
        "Total T-gate Count": resolved_summary.total_t_count,
        "Total Logical Depth": resolved_summary.total_depth,
        "Number of Algorithmic Qubits": resolved_summary.n_algorithmic_qubits,
        "Cultivation Error Rate": r.core_config.cultivation_error_rate,
        "V_CULT Factor": r.core_config.vcult_factor,
    }
    data.update(problem_specific_params)
    return data


def post_process_for_pec_runtime(
    sweep_results: List[SweepResult],
    *,
    time_per_surface_code_cycle: float = 1e-6,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Post-processes `run_sweep` output to calculate PEC runtime metrics.

    This function takes the list of `SweepResult` objects, resolves the
    symbolic expressions in each `FLASQSummary`, calculates the error-mitigated
    runtime using a Probabilistic Error Cancellation (PEC) model, and returns a
    pandas DataFrame of the final, enriched results.

    Args:
        sweep_results: The list of `SweepResult` objects returned by `run_sweep`.
        time_per_surface_code_cycle: The time for a single surface code cycle, used in
            the final runtime calculation. Defaults to 1 microsecond.
        n_jobs: Number of parallel jobs for ``joblib.Parallel``. Defaults to
            ``-1`` (use all available CPUs). Set to ``1`` to disable
            parallelism.

    Returns:
        A pandas DataFrame containing the fully processed and enriched results of the sweep.
    """
    parallel_gen = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(_process_single_result_for_pec)(r, time_per_surface_code_cycle)
        for r in sweep_results
    )
    processed_results = list(tqdm(parallel_gen, total=len(sweep_results), desc="Post-processing PEC results"))
    df = pd.DataFrame(processed_results)
    df = convert_sympy_exprs_in_df(df)
    return df


def post_process_for_failure_budget(
    sweep_results: List[SweepResult],
    error_budget: ErrorBudget,
    *,
    time_per_surface_code_cycle: float = 1e-6,
    error_prefactor: float = ERROR_PER_CYCLE_PREFACTOR,
) -> pd.DataFrame:
    """
    Post-processes `run_sweep` output to filter configurations based on a fixed failure budget.

    Calculates failure probabilities (P_fail) and filters results where:
    P_fail_Clifford <= error_budget.logical
    P_fail_T <= error_budget.cultivation

    Assumes sweep was run with total_allowable_rotation_error matching error_budget.synthesis.
    """
    processed_results = []

    filtered_results = [
        r
        for r in sweep_results
        if np.isclose(r.total_allowable_rotation_error, error_budget.synthesis)
    ]

    if not filtered_results:
        return pd.DataFrame()

    for r in filtered_results:
        flasq_model_obj, flasq_model_name = r.flasq_model_config
        t_react_val = r.reaction_time_in_cycles / r.core_config.code_distance

        assumptions = frozendict(
            {
                ROTATION_ERROR: r.logical_circuit_analysis[
                    "individual_allowable_rotation_error"
                ],
                V_CULT_FACTOR: r.core_config.vcult_factor,
                T_REACT: t_react_val,
            }
        )
        resolved_summary = r.flasq_summary.resolve_symbols(assumptions)

        lambda_val = 1e-2 / r.core_config.phys_error_rate

        try:
            P_fail_Clifford, P_fail_T = calculate_failure_probabilities(
                flasq_summary=resolved_summary,
                code_distance=r.core_config.code_distance,
                lambda_val=lambda_val,
                cultivation_error_rate=r.core_config.cultivation_error_rate,
                error_prefactor=error_prefactor,
            )
            P_fail_Clifford = float(P_fail_Clifford)
            P_fail_T = float(P_fail_T)
        except TypeError:
            continue

        if (
            P_fail_Clifford > error_budget.logical
            or P_fail_T > error_budget.cultivation
        ):
            continue

        try:
            wall_time = (
                time_per_surface_code_cycle
                * r.core_config.code_distance
                * float(resolved_summary.total_depth)
            )
        except TypeError:
            continue

        problem_specific_params = frozendict(
            {f"circuit_arg_{k}": v for k, v in r.circuit_builder_kwargs.items()}
            | (
                {
                    "circuit_arg_cultivation_data_source_distance": r.core_config.cultivation_data_source_distance,
                }
                if r.core_config.cultivation_data_source_distance is not None
                else {}
            )
        )
        data = {
            "Number of Physical Qubits": r.n_phys_qubits,
            "Physical Error Rate": r.core_config.phys_error_rate,
            "Code Distance": r.core_config.code_distance,
            "FLASQ Model": flasq_model_name.capitalize(),
            "Wall Clock Time (s)": wall_time,
            "P_fail_Clifford (P_log)": P_fail_Clifford,
            "P_fail_T (P_dis)": P_fail_T,
            "P_fail_Synthesis (P_syn)": error_budget.synthesis,
            "Sum of Failure Probabilities (P_log + P_dis)": P_fail_Clifford + P_fail_T,
            "Total T-gate Count": resolved_summary.total_t_count,
            "Total Logical Depth": resolved_summary.total_depth,
            "Number of Algorithmic Qubits": resolved_summary.n_algorithmic_qubits,
            "Number of Fluid Ancilla Qubits": resolved_summary.n_fluid_ancilla,
        }
        data.update(problem_specific_params)
        processed_results.append(data)

    if not processed_results:
        return pd.DataFrame()

    df = pd.DataFrame(processed_results)

    df = convert_sympy_exprs_in_df(df)

    return df
