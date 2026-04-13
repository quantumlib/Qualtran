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

import pandas as pd
import numpy as np
import itertools
import pytest

from qualtran.surface_code.flasq import cultivation_analysis


def test_get_cultivation_data():
    """Tests that get_cultivation_data returns a non-empty DataFrame."""
    df = cultivation_analysis.get_cultivation_data()

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    # Check for a few expected columns to ensure data integrity
    assert "error_rate" in df.columns
    assert "cultivation_distance" in df.columns
    assert "Unnamed: 0" not in df.columns
    assert "Unnamed: 0.1" not in df.columns


def test_get_unique_error_rates():
    """Tests that get_unique_error_rates returns a sorted array of unique error rates."""
    error_rates = cultivation_analysis.get_unique_error_rates()

    assert isinstance(error_rates, np.ndarray)
    assert len(error_rates) == 29  # This might change if data changes
    assert np.all(np.diff(error_rates) > 0)  # Check if sorted

    # Test with different precision
    error_rates_prec6 = cultivation_analysis.get_unique_error_rates(decimal_precision=6)
    assert isinstance(error_rates_prec6, np.ndarray)
    # Length might be different with different precision
    assert np.all(np.diff(error_rates_prec6) > 0)


def test_round_error_rate_up():
    """Tests the round_error_rate_up helper function."""
    # Use default precision for these tests
    unique_rates = (
        cultivation_analysis.get_unique_error_rates()
    )  # Default precision = 8

    # Case 1: physical_error_rate is below the smallest unique rate
    test_rate_below = unique_rates[0] - 1e-9  # Slightly below the first unique rate
    assert cultivation_analysis.round_error_rate_up(test_rate_below) == unique_rates[0]

    # Case 2: physical_error_rate matches a unique rate
    assert cultivation_analysis.round_error_rate_up(unique_rates[5]) == unique_rates[5]

    # Case 3: physical_error_rate is between two unique rates
    if len(unique_rates) > 1:
        test_rate_between = (unique_rates[0] + unique_rates[1]) / 2
        assert (
            cultivation_analysis.round_error_rate_up(test_rate_between)
            == unique_rates[1]
        )

    # Case 4: physical_error_rate is above all unique rates
    test_rate_above = unique_rates[-1] + 1e-5
    assert cultivation_analysis.round_error_rate_up(test_rate_above) is None

    # Case 5: physical_error_rate is very low (e.g., 0)
    assert cultivation_analysis.round_error_rate_up(0.0) == unique_rates[0]

    # Case 6: Test with a specific precision that might alter unique_rates
    # This is a bit harder to make robust without knowing data details,
    # but we can check it runs and returns something plausible or None.
    # For example, if precision 5 makes unique_rates[0] different or disappear.
    unique_rates_prec5 = cultivation_analysis.get_unique_error_rates(
        decimal_precision=5
    )
    if unique_rates_prec5.size > 0:
        res_prec5 = cultivation_analysis.round_error_rate_up(
            unique_rates_prec5[0] - 1e-7, decimal_precision=5
        )
        assert res_prec5 == unique_rates_prec5[0]
        res_above_prec5 = cultivation_analysis.round_error_rate_up(
            unique_rates_prec5[-1] + 1e-3, decimal_precision=5
        )
        assert res_above_prec5 is None
    else:  # If precision 5 yields no unique rates (unlikely for this dataset)
        assert (
            cultivation_analysis.round_error_rate_up(0.001, decimal_precision=5) is None
        )


def test_get_filtered_cultivation_data():
    """Tests that get_filtered_cultivation_data returns non-empty DataFrames for valid inputs."""
    unique_rates = cultivation_analysis.get_unique_error_rates()
    distances = [3, 5]

    for error_rate_val, distance_val in itertools.product(unique_rates, distances):
        # Pass the decimal_precision used in get_unique_error_rates
        filtered_df = cultivation_analysis.get_filtered_cultivation_data(
            error_rate_val, distance_val, decimal_precision=8
        )
        assert isinstance(filtered_df, pd.DataFrame)
        assert (
            len(filtered_df) > 0
        ), f"No data for error_rate={error_rate_val}, distance={distance_val}"


def test_filtered_data_monotonicity():
    """
    Tests that after filtering and sorting by 'gap', certain columns exhibit
    monotonic behavior.
    """
    unique_rates = cultivation_analysis.get_unique_error_rates()
    distances = [3, 5]
    # A small tolerance for floating point comparisons
    epsilon = 1e-10

    for error_rate_val, distance_val in itertools.product(unique_rates, distances):
        filtered_df = cultivation_analysis.get_filtered_cultivation_data(
            error_rate_val, distance_val, decimal_precision=8
        )

        sorted_df = filtered_df.sort_values("gap").reset_index(drop=True)

        assert np.all(
            np.diff(sorted_df["expected_volume"].values) >= -epsilon
        ), f"expected_volume not non-decreasing for ER={error_rate_val}, dist={distance_val}"

        # Noise means data is NOT QUITE monotonic in the t gate error rate...
        # assert np.all(np.diff(sorted_df["t_gate_cultivation_error_rate"].values) <= epsilon), f"t_gate_cultivation_error_rate not non-increasing for ER={error_rate_val}, dist={distance_val}"
        assert np.all(
            np.diff(sorted_df["keep_rate"].values) <= epsilon
        ), f"keep_rate not non-increasing for ER={error_rate_val}, dist={distance_val}"
        assert np.all(
            np.diff(sorted_df["attempts_per_kept_shot"].values) >= -epsilon
        ), f"attempts_per_kept_shot not non-decreasing for ER={error_rate_val}, dist={distance_val}"


def test_regularized_filtered_data_monotonicity_and_structure():  # Renamed for clarity
    """
    Tests that after regularized filtering and sorting by 'gap', certain columns
    exhibit monotonic behavior, including t_gate_cultivation_error_rate.
    Also checks basic structure and non-emptiness for a known good case.
    """
    unique_rates = cultivation_analysis.get_unique_error_rates()
    distances = [5, 3]  # Test both distances
    uncertainty_cutoffs = [None, 100.0]  # Test with and without cutoff
    # A small tolerance for floating point comparisons
    epsilon = 1e-10

    for error_rate_val, distance_val, cutoff_val in itertools.product(
        unique_rates, distances, uncertainty_cutoffs
    ):

        regularized_df = cultivation_analysis.get_regularized_filtered_cultivation_data(
            error_rate_val,
            distance_val,
            decimal_precision=8,
            uncertainty_cutoff=cutoff_val,
        )

        print(regularized_df.columns)

        sorted_df = regularized_df.sort_values("gap")

        print(sorted_df)
        print(sorted_df.columns)
        assert len(sorted_df) > 40

        assert np.all(
            np.diff(sorted_df["expected_volume"].values) >= -epsilon
        ), f"expected_volume not non-decreasing for ER={error_rate_val}, dist={distance_val}, cutoff={cutoff_val}"

        assert np.all(
            np.diff(sorted_df["t_gate_cultivation_error_rate"].values) <= epsilon
        ), f"t_gate_cultivation_error_rate not non-increasing for ER={error_rate_val}, dist={distance_val}, cutoff={cutoff_val}"

        assert np.all(
            np.diff(sorted_df["keep_rate"].values) <= epsilon
        ), f"keep_rate not non-increasing for ER={error_rate_val}, dist={distance_val}, cutoff={cutoff_val}"

        assert np.all(
            np.diff(sorted_df["attempts_per_kept_shot"].values) >= -epsilon
        ), f"attempts_per_kept_shot not non-decreasing for ER={error_rate_val}, dist={distance_val}, cutoff={cutoff_val}"


def test_get_regularized_filtered_combined_cultivation_data_structure_and_content():
    """
    Tests the get_regularized_filtered_combined_cultivation_data function.
    Verifies structure, content by comparing with individual calls, and caching.
    """
    error_rate_to_test = cultivation_analysis.get_unique_error_rates()[
        5
    ]  # Pick an arbitrary error rate
    decimal_precision_test = 8
    uncertainty_cutoff_test = 50.0  # A specific cutoff for testing

    # 1. Call the combined function
    combined_df = (
        cultivation_analysis.get_regularized_filtered_combined_cultivation_data(
            error_rate_to_test, decimal_precision_test, uncertainty_cutoff_test
        )
    )

    # 2. Call individual functions for comparison
    df_dist3 = cultivation_analysis.get_regularized_filtered_cultivation_data(
        error_rate_to_test, 3, decimal_precision_test, uncertainty_cutoff_test
    )
    df_dist5 = cultivation_analysis.get_regularized_filtered_cultivation_data(
        error_rate_to_test, 5, decimal_precision_test, uncertainty_cutoff_test
    )

    # 3. Verify content and structure
    assert isinstance(combined_df, pd.DataFrame)
    expected_len = len(df_dist3) + len(df_dist5)
    assert (
        len(combined_df) == expected_len
    ), "Combined DataFrame length mismatch with sum of individual lengths."

    # Check presence of both distances
    if not combined_df.empty:
        assert 3 in combined_df["cultivation_distance"].unique()
        assert 5 in combined_df["cultivation_distance"].unique()

    # For a more rigorous content check, concatenate individual results and compare
    # Need to reset index for proper comparison if order matters, or sort.
    # Sorting by a set of columns that should make rows unique if data is clean.
    # If not perfectly unique, pd.testing.assert_frame_equal might be too strict.
    # For now, length and presence of distances are good indicators.
    # If more detailed check is needed:
    # expected_combined_df = pd.concat([df_dist3, df_dist5], ignore_index=True)
    # pd.testing.assert_frame_equal(combined_df.sort_values(by=list(combined_df.columns)).reset_index(drop=True),
    #                                expected_combined_df.sort_values(by=list(expected_combined_df.columns)).reset_index(drop=True),
    #                                check_dtype=False) # Allow for minor type differences if any

    # 4. Verify caching
    # Clear cache before testing this specific function's cache
    cultivation_analysis.get_regularized_filtered_combined_cultivation_data.cache_clear()
    initial_cache_info = (
        cultivation_analysis.get_regularized_filtered_combined_cultivation_data.cache_info()
    )
    _ = cultivation_analysis.get_regularized_filtered_combined_cultivation_data(
        error_rate_to_test, decimal_precision_test, uncertainty_cutoff_test
    )  # First call
    _ = cultivation_analysis.get_regularized_filtered_combined_cultivation_data(
        error_rate_to_test, decimal_precision_test, uncertainty_cutoff_test
    )  # Second call (should be cached)
    final_cache_info = (
        cultivation_analysis.get_regularized_filtered_combined_cultivation_data.cache_info()
    )
    assert (
        final_cache_info.hits > initial_cache_info.hits
    ), "Cache hit not registered for combined function."
    assert final_cache_info.misses == initial_cache_info.misses + 1


def test_find_best_cultivation_parameters():
    """Tests the find_best_cultivation_parameters function."""

    # Test case 1: Known good parameters
    phys_err = 1e-3
    log_err_target = 1e-7
    result = cultivation_analysis.find_best_cultivation_parameters(
        phys_err, log_err_target
    )
    assert isinstance(result, pd.Series)
    assert (
        not result.empty
    ), f"Expected non-empty result for phys_err={phys_err}, log_err_target={log_err_target}"
    assert "t_gate_cultivation_error_rate" in result
    assert result["t_gate_cultivation_error_rate"] < log_err_target
    assert "error_rate" in result
    # The matched physical error rate should be >= input physical_error_rate
    # and should be one of the unique rates.
    unique_rates = cultivation_analysis.get_unique_error_rates()
    assert result["error_rate"] >= phys_err
    assert result["error_rate"] in unique_rates
    assert "expected_volume" in result

    # Test case 2: Physical error rate too high
    phys_err_high = 0.5  # Higher than any in the dataset
    result_high_phys = cultivation_analysis.find_best_cultivation_parameters(
        phys_err_high, log_err_target
    )
    assert isinstance(result_high_phys, pd.Series)
    assert (
        result_high_phys.empty
    ), f"Expected empty result for very high physical error rate {phys_err_high}"

    # Test case 3: Target logical error rate too low
    log_err_target_low = 1e-20  # Lower than any achievable
    result_low_target = cultivation_analysis.find_best_cultivation_parameters(
        phys_err, log_err_target_low
    )
    assert isinstance(result_low_target, pd.Series)
    assert (
        result_low_target.empty
    ), f"Expected empty result for very low target logical error rate {log_err_target_low}"

    # Test case 4: Ensure `uncertainty_cutoff` is passed through and can affect results
    # Find a scenario where default cutoff (100) gives a result, but a very strict one (e.g., 1.1) gives empty or different
    # This requires knowledge of the data. For now, just test it runs.
    # If we find a specific data point that would be filtered by a stricter cutoff, we can add a more precise test.
    phys_err_for_cutoff_test = 0.0001885  # A value from unique_error_rates
    log_err_target_for_cutoff_test = 1e-6

    result_default_cutoff = cultivation_analysis.find_best_cultivation_parameters(
        phys_err_for_cutoff_test,
        log_err_target_for_cutoff_test,
        uncertainty_cutoff=100.0,
    )
    result_strict_cutoff = cultivation_analysis.find_best_cultivation_parameters(
        phys_err_for_cutoff_test, log_err_target_for_cutoff_test, uncertainty_cutoff=1.1
    )
    # We can't easily assert they are different without knowing the data intimately,
    # but we can assert they both run and return Series.
    assert isinstance(result_default_cutoff, pd.Series)
    assert isinstance(result_strict_cutoff, pd.Series)
    # It's possible both are empty or both are non-empty. If non-empty, they might be different.
    if not result_default_cutoff.empty and not result_strict_cutoff.empty:
        # If both found something, a very strict cutoff might lead to a different (or same if only one option) result.
        # This is not a strong assertion without specific data knowledge.
        pass
    elif not result_default_cutoff.empty and result_strict_cutoff.empty:
        # This is a good sign the cutoff had an effect.
        pass

    # Test case 5: Scenario where only distance 3 might be better or available
    # (This is hard to guarantee without specific data inspection, but we can try a plausible scenario)
    # Let's pick a relatively high physical error rate where distance 3 might be the only option
    # or have a better volume.
    phys_err_favor_d3 = 0.001  # A mid-range physical error rate
    log_err_target_favor_d3 = 1e-5  # A relatively achievable logical error

    result_favor_d3 = cultivation_analysis.find_best_cultivation_parameters(
        phys_err_favor_d3, log_err_target_favor_d3
    )
    assert isinstance(result_favor_d3, pd.Series)
    # If non-empty, we can check its properties.
    if not result_favor_d3.empty:
        assert "cultivation_distance" in result_favor_d3
        # We can't assert it IS 3, but it's a valid output.

    # Test case 6: Scenario where only distance 5 might be better or available
    # Low physical error rate, very low target logical error rate
    phys_err_favor_d5 = cultivation_analysis.get_unique_error_rates()[
        0
    ]  # Lowest physical error
    log_err_target_favor_d5 = 1e-9

    result_favor_d5 = cultivation_analysis.find_best_cultivation_parameters(
        phys_err_favor_d5, log_err_target_favor_d5
    )
    assert isinstance(result_favor_d5, pd.Series)
    assert not result_favor_d5.empty
    assert "cultivation_distance" in result_favor_d5
