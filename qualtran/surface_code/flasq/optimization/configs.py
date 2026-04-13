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
from typing import Any, Iterable, List, NamedTuple, Optional, Union

from frozendict import frozendict
from qualtran.surface_code.flasq import cultivation_analysis


class CoreParametersConfig(NamedTuple):
    """Core interdependent parameters for a single sweep point.

    Attributes:
        code_distance: Surface code distance d.
        phys_error_rate: p_phys in the paper. Physical error rate.
        cultivation_error_rate: p_mag in the paper. Per-T-gate error rate
            from the cultivation protocol.
        vcult_factor: Normalized cultivation volume: v(p_phys, p_cult)
            divided by block size (2*(d+1)^2 * d).
        cultivation_data_source_distance: Which cultivation distance
            (3 or 5) supplied the data for this config, or None.
        target_t_error: Target per-T-gate error used to look up
            cultivation data, if applicable.
    """

    code_distance: int
    phys_error_rate: float
    cultivation_error_rate: float
    vcult_factor: float
    cultivation_data_source_distance: Optional[int] = None
    target_t_error: Optional[float] = None

    @classmethod
    def from_cultivation_analysis(
        cls,
        physical_error_rate: float,
        target_individual_t_gate_error: float,
        reference_code_distance: int,
    ) -> "CoreParametersConfig":
        """Creates a config from cultivation simulation data.

        Looks up the cultivation row that minimises expected volume while
        keeping p_cult below ``target_individual_t_gate_error``, then derives
        ``vcult_factor`` at ``reference_code_distance``.

        In workflows that don't sweep code distance (e.g., HWP),
        ``reference_code_distance`` becomes the effective code distance for
        all downstream calculations.

        Args:
            physical_error_rate: p_phys.
            target_individual_t_gate_error: Upper bound on p_cult per T gate.
            reference_code_distance: Fixed code distance for deriving
                vcult_factor.
        """
        best_cult_params = cultivation_analysis.find_best_cultivation_parameters(
            physical_error_rate=physical_error_rate,
            target_logical_error_rate=target_individual_t_gate_error,
        )
        if best_cult_params.empty:
            raise ValueError(
                "No suitable cultivation parameters found for the given inputs."
            )

        vcult_factor = best_cult_params["expected_volume"] / (
            2 * (reference_code_distance + 1) ** 2 * reference_code_distance
        )

        return CoreParametersConfig(
            code_distance=reference_code_distance,
            phys_error_rate=physical_error_rate,
            cultivation_error_rate=best_cult_params["t_gate_cultivation_error_rate"],
            vcult_factor=vcult_factor,
            cultivation_data_source_distance=best_cult_params["cultivation_distance"],
            target_t_error=target_individual_t_gate_error,
        )


def _ensure_iterable(
    value: Any, treat_frozendict_as_single_item: bool = False
) -> Iterable[Any]:
    """Converts a single value to a single-element list if it's not already iterable.

    Special handling for strings and optionally for frozendict.
    """
    if treat_frozendict_as_single_item and isinstance(value, frozendict):
        return [value]
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return [value]
    if isinstance(value, str):
        return [value]
    if not isinstance(value, Iterable):
        return [value]
    return value


def generate_configs_for_specific_cultivation_assumptions(
    code_distance_list: Union[int, Iterable[int]],
    phys_error_rate_list: Union[float, Iterable[float]],
    cultivation_error_rate: float,
    vcult_factor: float,
) -> List[CoreParametersConfig]:
    """
    Generates a list of CoreParametersConfig objects for fixed cultivation
    error rate and V_CULT_FACTOR, sweeping over code distances and physical error rates.

    Args:
        code_distance_list: Single value or iterable of code distances.
        phys_error_rate_list: Single value or iterable of physical error rates.
        cultivation_error_rate: The fixed cultivation error rate to use.
        vcult_factor: The fixed V_CULT_FACTOR to use.

    Returns:
        A list of CoreParametersConfig objects.
    """
    configs = []
    d_list = list(_ensure_iterable(code_distance_list))
    p_err_list = list(_ensure_iterable(phys_error_rate_list))

    for d_val, p_err_val in itertools.product(d_list, p_err_list):
        configs.append(
            CoreParametersConfig(
                code_distance=d_val,
                phys_error_rate=p_err_val,
                cultivation_error_rate=cultivation_error_rate,
                vcult_factor=vcult_factor,
                cultivation_data_source_distance=None,
            )
        )
    return configs


def generate_configs_from_cultivation_data(
    code_distance_list: Union[int, Iterable[int]],
    phys_error_rate_list: Union[float, Iterable[float]],
    cultivation_data_source_distance_list: Union[int, Iterable[int], None] = None,
    cultivation_data_decimal_precision: int = 8,
    cultivation_data_slack_factor: float = 0.995,
    cultivation_data_uncertainty_cutoff: Optional[
        float
    ] = 100,  # Default value from cultivation_analysis
    cultivation_data_sampling_frequency: Optional[int] = None,
    round_error_rate_up_to_simulated_cultivation_data: bool = True,
) -> List[CoreParametersConfig]:
    """
    Generates CoreParametersConfig objects by deriving cultivation_error_rate and
    vcult_factor from cultivation analysis data.

    Args:
        code_distance_list: Single or iterable of code distances.
        phys_error_rate_list: Single or iterable of physical error rates.
        cultivation_data_source_distance_list: Single or iterable of cultivation
            distances (e.g., 3, 5) to fetch data for. Defaults to [3, 5].
        cultivation_data_decimal_precision: Precision for cultivation data fetching.
        cultivation_data_slack_factor: Empirical tolerance for matching physical
            error rates to the discrete set available in the cultivation simulation
            data. Slightly below 1.0 to handle floating-point rounding: ensures
            that a queried p_phys nominally equal to a tabulated value is not
            accidentally rounded above it and mapped to the next-higher entry.
            The exact value (0.995) is a conservative default with no theoretical
            significance.
        cultivation_data_uncertainty_cutoff: Empirical cutoff on the ratio
            ``high_10 / low_10``. Rows where the 10× confidence interval spans
            more than this factor are discarded as insufficiently converged
            simulation data. The default of 100 is a pragmatic choice: tight
            enough to exclude points with very few non-discarded shots (where
            binomial statistics are unreliable), loose enough to retain all data
            points in the regime of practical interest. No sensitivity to this
            value was observed in the final resource estimates.
        cultivation_data_sampling_frequency: Optional. If None, all valid rows from
            the regularized cultivation data (cult_df) are used. If an integer K > 0,
            samples every K-th row from cult_df, starting from the tail (last row).
            For example, K=1 takes all rows (in order: last, second to last, ...).
            K >= len(cult_df) takes only the last row.
            Must be None or a positive integer.
        round_error_rate_up_to_simulated_cultivation_data: If True, each physical_error_rate
            from phys_error_rate_list will be rounded up to the nearest available error rate
            in the cultivation dataset before fetching data.

    Returns:
        A list of CoreParametersConfig objects.
    """
    configs = []
    d_list = list(_ensure_iterable(code_distance_list))
    p_err_list = list(_ensure_iterable(phys_error_rate_list))
    source_dist_list = (
        list(_ensure_iterable(cultivation_data_source_distance_list))
        if cultivation_data_source_distance_list is not None
        else [3, 5]
    )

    for current_d, current_phys_err, current_cult_data_dist in itertools.product(
        d_list, p_err_list, source_dist_list
    ):
        phys_err_for_cult_lookup = current_phys_err
        if round_error_rate_up_to_simulated_cultivation_data:
            rounded_err = cultivation_analysis.round_error_rate_up(
                current_phys_err,
                cultivation_data_decimal_precision,
                slack_factor=cultivation_data_slack_factor,
            )
            if rounded_err is None:
                # No suitable error rate in cultivation data for this current_phys_err
                continue
            phys_err_for_cult_lookup = rounded_err

        cult_df = cultivation_analysis.get_regularized_filtered_cultivation_data(
            error_rate=phys_err_for_cult_lookup,  # Use potentially rounded error rate
            cultivation_distance=current_cult_data_dist,
            decimal_precision=cultivation_data_decimal_precision,
            uncertainty_cutoff=cultivation_data_uncertainty_cutoff,
        )

        if cult_df.empty:
            continue

        if cultivation_data_sampling_frequency is None:
            rows_to_process = cult_df
        else:
            k = cultivation_data_sampling_frequency
            if k <= 0:
                raise ValueError(
                    "cultivation_data_sampling_frequency must be None or a positive integer."
                )
            if not cult_df.empty:
                # Sample every k-th row from the tail of the already sorted/filtered cult_df
                indices = range(len(cult_df) - 1, -1, -k)
                rows_to_process = cult_df.iloc[list(indices)]
            else:
                rows_to_process = cult_df  # empty
        for _, row in rows_to_process.iterrows():
            derived_cult_error_rate = row["t_gate_cultivation_error_rate"]
            expected_volume = row["expected_volume"]

            derived_vcult_factor = expected_volume / (
                2 * (current_d + 1) ** 2 * current_d
            )
            configs.append(
                CoreParametersConfig(
                    code_distance=current_d,
                    phys_error_rate=current_phys_err,
                    cultivation_error_rate=derived_cult_error_rate,
                    vcult_factor=derived_vcult_factor,
                    cultivation_data_source_distance=current_cult_data_dist,
                )
            )
    return configs


class ErrorBudget(NamedTuple):
    """Allocation of total error budget across QEC failure modes.

    Attributes:
        logical: epsilon_log. Clifford/memory failure budget.
        cultivation: epsilon_cult. T-gate cultivation failure budget.
        synthesis: epsilon_syn. Rotation approximation bias budget.
    """

    logical: float
    cultivation: float
    synthesis: float

    @property
    def total(self) -> float:
        return self.logical + self.cultivation + self.synthesis
