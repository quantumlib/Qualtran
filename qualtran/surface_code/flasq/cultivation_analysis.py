"""Cultivation data lookup for magic state preparation.

Loads pre-computed cultivation simulation data from CSV and finds optimal
cultivation parameters (error rate, expected volume) for a given physical
error rate and target logical error rate.

Key CSV columns and paper terminology:
    error_rate: p_phys in the paper. Physical error rate.
    t_gate_cultivation_error_rate: p_cult in the paper. Logical error rate
        per T gate from the cultivation protocol.
    expected_volume: v(p_phys, p_cult) in the paper. Cultivation spacetime
        volume in physical qubits × cycles.
    cultivation_distance: Code distance for the cultivation protocol (3 or 5).
        These are the exhaustive set of distances with published simulation
        results.
"""

import pandas as pd
import numpy as np
import importlib.resources
from typing import Optional
from functools import lru_cache


@lru_cache(maxsize=1)
def get_cultivation_data() -> pd.DataFrame:
    """Loads the cultivation simulation summary data from the CSV file."""
    # Use importlib.resources to get a path to the data file
    # This assumes your package is named 'qualtran_flasq' and the data
    # is in a 'data' subdirectory within that package.
    data_file_path = importlib.resources.files("qualtran_flasq.data").joinpath(
        "cultivation_simulation_summary.csv"
    )
    columns_to_drop = ["Unnamed: 0", "Unnamed: 0.1"]
    with importlib.resources.as_file(data_file_path) as csv_path:
        df = pd.read_csv(csv_path)
        # Drop the specified columns if they exist
        df.drop(
            columns=[col for col in columns_to_drop if col in df.columns],
            inplace=True,
            errors="ignore",
        )
    return df


def get_unique_error_rates(decimal_precision: int = 8) -> np.ndarray:
    """
    Loads the cultivation data and returns a sorted NumPy array of unique
    error rates, rounded to a specified decimal precision.
    """
    unique_error_rates_arr = np.sort(
        get_cultivation_data()["error_rate"].round(decimal_precision).unique()
    )
    return unique_error_rates_arr


def round_error_rate_up(
    physical_error_rate: float, decimal_precision: int = 8, slack_factor: float = 1.0
) -> Optional[float]:
    """
    Finds the smallest unique error rate from the dataset that is greater than
    or equal to the given physical_error_rate. Unique error rates are
    determined by rounding values in the dataset to the specified decimal_precision.

    Args:
        physical_error_rate: The target physical error rate.
        decimal_precision: The decimal precision used to determine unique error rates.
        slack_factor: Empirical tolerance for matching physical error rates to
            the discrete set available in the simulation data. Values slightly
            below 1.0 prevent floating-point rounding from pushing a queried
            rate above a nominally equal tabulated value.

    Returns:
        The smallest unique error rate >= physical_error_rate * slack_factor, or None if no such
        rate exists (e.g., if physical_error_rate*slack_factor is higher than all unique rates).
    """
    available_error_rates = get_unique_error_rates(decimal_precision)
    suitable_rates = available_error_rates[
        available_error_rates >= physical_error_rate * slack_factor
    ]

    if suitable_rates.size == 0:
        return None
    return float(suitable_rates.min())


def get_filtered_cultivation_data(
    error_rate: float, cultivation_distance: int, decimal_precision: int = 8
) -> pd.DataFrame:
    """
    Filters the cultivation data for a specific error_rate (rounded) and cultivation_distance.

    Args:
        error_rate: The target error rate.
        cultivation_distance: The target cultivation distance (e.g., 3 or 5).
        decimal_precision: The decimal precision to use for rounding the error_rate
                           column for comparison.

    Returns:
        A pandas DataFrame view containing the filtered rows.
    """
    df = get_cultivation_data()
    # Round the 'error_rate' column in the DataFrame for comparison
    # and create a boolean Series for the error rate condition
    error_condition = np.isclose(
        df["error_rate"],
        error_rate,
        atol=10 ** (-decimal_precision - 1),
    )

    # Create a boolean Series for the cultivation distance condition
    distance_condition = (
        df["cultivation_distance"] == cultivation_distance
    )

    # Combine the conditions
    combined_condition = error_condition & distance_condition

    # Return a view of the DataFrame
    return df[combined_condition]


@lru_cache(maxsize=None)
def get_regularized_filtered_cultivation_data(
    error_rate: float,
    cultivation_distance: int,
    decimal_precision: int = 8,
    uncertainty_cutoff: Optional[float] = 100,  # Default value added
) -> pd.DataFrame:
    """Filters and regularizes cultivation data.

    Steps:
        1. Applies initial filtering using ``get_filtered_cultivation_data``.
        2. Drops data points where ``t_gate_cultivation_error_rate`` (p_cult) is zero.
        3. If ``uncertainty_cutoff`` is provided, drops points where
           ``high_10 / low_10`` exceeds the cutoff. This is an empirical
           threshold for removing noisy simulation points.
        4. Sorts remaining data by 'gap' and enforces monotonicity:
           ``t_gate_cultivation_error_rate`` must be non-increasing.
           This data-cleaning step compensates for finite simulation
           samples — with infinite data p_cult would be monotonic.

    Args:
        error_rate: The target error rate (p_phys) for initial filtering.
        cultivation_distance: The cultivation protocol distance (3 or 5).
        decimal_precision: Decimal precision for error rate comparison.
        uncertainty_cutoff: Empirical cutoff on the ratio
            ``high_10 / low_10``. Rows exceeding this factor are discarded
            as insufficiently converged simulation data. The default of 100
            is a pragmatic choice with no observed sensitivity in final
            resource estimates.

    Returns:
        A pandas DataFrame containing the regularized and filtered data.
    """
    df = get_cultivation_data()
    filtered_df = get_filtered_cultivation_data(
        error_rate, cultivation_distance, decimal_precision
    )

    if filtered_df.empty:
        return filtered_df  # Returns an empty view/copy from the original

    current_indices = filtered_df.index

    # 1. Filter out zero t_gate_cultivation_error_rate
    # Accessing the original dataframe with current_indices to check the condition
    condition_non_zero_t_error = (
        df.loc[current_indices, "t_gate_cultivation_error_rate"] != 0
    )
    current_indices = current_indices[condition_non_zero_t_error]

    if not current_indices.any():  # Check if Index is empty
        return df.loc[current_indices]

    # 2. Filter by uncertainty_cutoff (Optional)
    if uncertainty_cutoff is not None:
        data_for_uncertainty_check = df.loc[current_indices]
        # Calculate ratio, pandas handles division by zero (inf/nan)
        # Ensure low_10 is not zero before division to avoid warnings and inf values if not desired
        # We will keep inf values if low_10 is 0 and high_10 is not, as these should be filtered by cutoff.
        # If low_10 is 0 and high_10 is 0, ratio is nan, which is fine.
        with np.errstate(
            divide="ignore", invalid="ignore"
        ):  # Suppress division by zero warnings for this block
            ratio = (
                data_for_uncertainty_check["high_10"]
                / data_for_uncertainty_check["low_10"]
            )

        # Keep rows where ratio is less than or equal to cutoff
        # np.nan <= cutoff is False, np.inf <= cutoff is False (unless cutoff is inf)
        condition_uncertainty_ok = ratio <= uncertainty_cutoff
        current_indices = current_indices[condition_uncertainty_ok]

        if not current_indices.any():
            return df.loc[current_indices]

    # 3. Monotonicity Filter for t_gate_cultivation_error_rate
    data_for_monotonic_check = df.loc[current_indices]

    if data_for_monotonic_check.empty:
        return data_for_monotonic_check  # Already a view of original via .loc

    sorted_data = data_for_monotonic_check.sort_values("gap")

    # Vectorized monotonicity filter
    mask = (
        sorted_data["t_gate_cultivation_error_rate"]
        == sorted_data["t_gate_cultivation_error_rate"].cummin()
    )
    current_indices = sorted_data[mask].index

    return df.loc[current_indices]


@lru_cache(maxsize=None)
def get_regularized_filtered_combined_cultivation_data(
    error_rate: float,
    decimal_precision: int = 8,
    uncertainty_cutoff: Optional[float] = 100,
) -> pd.DataFrame:
    """
    Retrieves and combines regularized cultivation data for distances 3 and 5.

    This function calls `get_regularized_filtered_cultivation_data` for
    cultivation_distance 3 and 5 separately and concatenates the results.
    The combined DataFrame is cached.

    Args:
        error_rate: The target error rate for initial filtering.
        decimal_precision: Decimal precision for error rate comparison.
        uncertainty_cutoff: Optional. Cutoff for the ratio `high_10 / low_10`.

    Returns:
        A pandas DataFrame containing the combined, regularized data for
        cultivation distances 3 and 5.
    """
    df_dist3 = get_regularized_filtered_cultivation_data(
        error_rate, 3, decimal_precision, uncertainty_cutoff
    )
    df_dist5 = get_regularized_filtered_cultivation_data(
        error_rate, 5, decimal_precision, uncertainty_cutoff
    )

    combined_df = pd.concat([df_dist3, df_dist5], ignore_index=True)
    return combined_df


def find_best_cultivation_parameters(
    physical_error_rate: float,
    target_logical_error_rate: float,
    decimal_precision: int = 8,
    uncertainty_cutoff: Optional[float] = 100,
) -> pd.Series:
    """Finds cultivation parameters that minimize expected volume.

    Selects the cultivation row (distance 3 or 5) whose
    ``t_gate_cultivation_error_rate`` (p_cult) is below
    ``target_logical_error_rate`` and whose ``expected_volume``
    (v(p_phys, p_cult)) is minimal. When both distances yield valid
    rows, the one with lower expected volume wins.

    Args:
        physical_error_rate: p_phys. Rounded up to the nearest available
            simulation error rate.
        target_logical_error_rate: Upper bound on p_cult.
        decimal_precision: Precision for matching physical_error_rate.
        uncertainty_cutoff: Empirical cutoff on the ``high_10 / low_10``
            ratio during regularization. Rows exceeding this factor are
            discarded as insufficiently converged. The default of 100 is a
            pragmatic choice with no observed sensitivity in final estimates.

    Returns:
        A pandas Series representing the best row, or an empty Series if no
        suitable parameters are found.
    """
    # Get unique error rates, which are already rounded to decimal_precision
    current_unique_error_rates = get_unique_error_rates(decimal_precision)

    # Find the smallest rate in unique_error_rates that is >= physical_error_rate.
    # This effectively "rounds up" physical_error_rate to the nearest available unique rate.
    suitable_rates = current_unique_error_rates[
        current_unique_error_rates >= physical_error_rate
    ]
    if suitable_rates.size == 0:
        # No unique rate is >= physical_error_rate
        return pd.Series(dtype=float)
    matched_physical_error_rate = suitable_rates.min()

    best_row_dist3: Optional[pd.Series] = None
    best_row_dist5: Optional[pd.Series] = None

    for dist in [3, 5]:
        reg_df = get_regularized_filtered_cultivation_data(
            error_rate=matched_physical_error_rate,  # Use the matched rate
            cultivation_distance=dist,
            decimal_precision=decimal_precision,
            uncertainty_cutoff=uncertainty_cutoff,
        )

        if not reg_df.empty:
            # Filter for t_gate_cultivation_error_rate < target_logical_error_rate
            filtered_by_target = reg_df[
                reg_df["t_gate_cultivation_error_rate"] < target_logical_error_rate
            ]

            if not filtered_by_target.empty:
                # Select the row with the largest t_gate_cultivation_error_rate from this subset
                # (i.e., the best one that still meets the target)
                best_row_for_dist = filtered_by_target.loc[
                    filtered_by_target["t_gate_cultivation_error_rate"].idxmax()
                ]
                if dist == 3:
                    best_row_dist3 = best_row_for_dist
                else:  # dist == 5
                    best_row_dist5 = best_row_for_dist

    # Determine final result based on findings
    if best_row_dist3 is None and best_row_dist5 is None:
        return pd.Series(dtype=float)  # Case 1
    elif best_row_dist3 is not None and best_row_dist5 is None:
        return best_row_dist3  # Case 2 (only dist 3 found)
    elif best_row_dist3 is None and best_row_dist5 is not None:
        return best_row_dist5  # Case 2 (only dist 5 found)
    else:  # Both best_row_dist3 and best_row_dist5 are not None (Case 3)
        # Explicitly cast to avoid potential type errors with mypy if Series were None
        # Although the logic above ensures they are not None here.
        best_row_dist3_series = pd.Series(best_row_dist3)  # Ensure it's a Series
        best_row_dist5_series = pd.Series(best_row_dist5)  # Ensure it's a Series

        if (
            best_row_dist3_series["expected_volume"]
            <= best_row_dist5_series["expected_volume"]
        ):
            return best_row_dist3_series
        else:
            return best_row_dist5_series
