import pytest
import sympy
from unittest.mock import MagicMock

# (Assuming standard imports for FLASQSummary, etc.)
from qualtran.surface_code.flasq.flasq_model import FLASQSummary
from qualtran.surface_code.flasq.error_mitigation import (
    calculate_failure_probabilities,
    ERROR_PER_CYCLE_PREFACTOR,
)


# Helper to create a mock FLASQSummary for failure calculation tests
# We use MagicMock here as FLASQSummary is complex to instantiate fully if only a few fields are needed.
def _mock_summary_for_failures(
    total_spacetime_volume, total_t_count, cultivation_volume=0
):
    summary = MagicMock(spec=FLASQSummary)
    summary.total_spacetime_volume = total_spacetime_volume
    summary.total_t_count = total_t_count
    summary.cultivation_volume = cultivation_volume
    summary.regular_spacetime_volume = total_spacetime_volume - cultivation_volume
    return summary


class FailureProbabilitiesTestSuite:
    def test_calculate_failure_probabilities_basic(self):
        """Test Case 1: Basic calculation with hand-verified numbers."""
        # Setup:
        # d=9, Lambda=10, c_cyc=0.03. p_cyc = 3e-7.
        # Spacetime volume=1000. r=9. Total cycles = 9000.
        # M=500. p_mag=1e-8.

        summary = _mock_summary_for_failures(
            total_spacetime_volume=1000, total_t_count=500
        )
        d = 9
        lambda_val = 10
        p_mag = 1e-8

        P_cliff, P_t = calculate_failure_probabilities(
            flasq_summary=summary,
            code_distance=d,
            lambda_val=lambda_val,
            cultivation_error_rate=p_mag,
            error_prefactor=ERROR_PER_CYCLE_PREFACTOR,
        )

        # Expected results
        # p_cyc = 0.03 * 10**-5 = 3e-7. Cycles = 1000 * 9 = 9000.
        # P_cliff = 1 - (1 - 3e-7)**9000 ≈ 0.00269635
        # P_t = 1 - (1 - 1e-8)**500 ≈ 4.9999875e-6
        expected_P_cliff = 1 - (1 - 3e-7) ** 9000
        expected_P_t = 1 - (1 - 1e-8) ** 500

        assert pytest.approx(float(P_cliff)) == expected_P_cliff
        assert pytest.approx(float(P_t)) == expected_P_t

    def test_calculate_failure_probabilities_symbolic(self):
        """Test Case 2: Symbolic inputs."""
        C_sym = sympy.Symbol("C_total")
        M_sym = sympy.Symbol("M")
        d_sym = sympy.Symbol("d")
        L_sym = sympy.Symbol("Lambda")
        P_mag_sym = sympy.Symbol("P_mag")

        # For this test, cultivation_volume is implicitly 0 in the mock
        summary = _mock_summary_for_failures(
            total_spacetime_volume=C_sym, total_t_count=M_sym
        )

        P_cliff, P_t = calculate_failure_probabilities(
            flasq_summary=summary,
            code_distance=d_sym,
            lambda_val=L_sym,
            cultivation_error_rate=P_mag_sym,
            error_prefactor=0.1,  # Using a specific prefactor for the test
        )

        # Expected symbolic expressions
        expected_p_cyc = 0.1 * L_sym ** (-(d_sym + 1) / 2)
        # Cycles = (C_sym - 0) * d_sym
        expected_P_cliff = 1 - (1 - expected_p_cyc) ** (C_sym * d_sym)
        expected_P_t = 1 - (1 - P_mag_sym) ** M_sym

        # Use sympy.simplify for robust comparison of symbolic expressions
        assert sympy.simplify(P_cliff - expected_P_cliff) == 0
        assert sympy.simplify(P_t - expected_P_t) == 0


class ErrorMitigationMetricsTestSuite:
    """Characterization tests for calculate_error_mitigation_metrics.

    This function was previously tested only indirectly (through ising_test.py
    integration tests). These tests lock its behavior at the unit level.
    """

    @staticmethod
    def _mock_summary_for_metrics(
        total_spacetime_volume,
        total_t_count,
        cultivation_volume,
        total_depth,
    ):
        summary = MagicMock(spec=FLASQSummary)
        summary.total_spacetime_volume = total_spacetime_volume
        summary.total_t_count = total_t_count
        summary.cultivation_volume = cultivation_volume
        summary.total_depth = total_depth
        summary.regular_spacetime_volume = total_spacetime_volume - cultivation_volume
        return summary

    def test_calculate_metrics_basic_concrete(self):
        """Hand-verified concrete calculation of error mitigation metrics."""
        from qualtran.surface_code.flasq.error_mitigation import calculate_error_mitigation_metrics

        summary = self._mock_summary_for_metrics(
            total_spacetime_volume=1000.0,
            total_t_count=200.0,
            cultivation_volume=100.0,
            total_depth=50.0,
        )

        eff_time, wall_time, gamma_block = calculate_error_mitigation_metrics(
            flasq_summary=summary,
            time_per_surface_code_cycle=1e-6,
            code_distance=9,
            lambda_val=100.0,  # 1e-2 / 1e-4
            cultivation_error_rate=1e-8,
        )

        # Wall clock time = time_per_cycle * code_distance * total_depth
        # = 1e-6 * 9 * 50 = 4.5e-4
        expected_wall_time = 1e-6 * 9 * 50.0
        assert pytest.approx(float(wall_time), rel=1e-10) == expected_wall_time

        # p_cyc = 0.03 * 100^(-(9+1)/2) = 0.03 * 100^-5 = 0.03 * 1e-10 = 3e-12
        p_cyc = 0.03 * 100.0 ** (-5.0)

        # gamma_per_cycle = ((1-p_cyc)^-1)^2 (symmetric X/Z)
        gamma_z = (1 - p_cyc) ** -1
        gamma_x = (1 - p_cyc) ** -1
        gamma_per_cycle = gamma_z * gamma_x

        # gamma_per_t = (1 - 2*1e-8)^-1
        gamma_per_t = (1 - 2 * 1e-8) ** -1

        # regular_spacetime_cycles = (1000 - 100) * 9 = 8100
        import math

        regular_cycles = (1000.0 - 100.0) * 9
        log_gamma = (
            math.log(gamma_per_cycle) * regular_cycles
            + math.log(gamma_per_t) * 200.0
        )
        gamma_circuit = math.exp(log_gamma)
        sampling_overhead = gamma_circuit**2
        expected_eff_time = expected_wall_time * sampling_overhead

        assert pytest.approx(float(eff_time), rel=1e-6) == expected_eff_time

        # gamma_per_block = gamma_per_cycle^code_distance
        expected_gamma_block = gamma_per_cycle**9
        assert pytest.approx(float(gamma_block), rel=1e-10) == expected_gamma_block

    def test_calculate_metrics_zero_t_count(self):
        """When there are no T gates, cultivation-related overhead should be minimal."""
        from qualtran.surface_code.flasq.error_mitigation import calculate_error_mitigation_metrics

        summary = self._mock_summary_for_metrics(
            total_spacetime_volume=500.0,
            total_t_count=0.0,
            cultivation_volume=0.0,
            total_depth=25.0,
        )

        eff_time, wall_time, gamma_block = calculate_error_mitigation_metrics(
            flasq_summary=summary,
            time_per_surface_code_cycle=1e-6,
            code_distance=7,
            lambda_val=100.0,
            cultivation_error_rate=1e-8,
        )

        # Wall clock time = 1e-6 * 7 * 25 = 1.75e-4
        assert pytest.approx(float(wall_time), rel=1e-10) == 1.75e-4

        # With zero T gates, log_gamma has only the Clifford contribution
        # and effective time should still be >= wall clock time
        assert float(eff_time) >= float(wall_time)

    def test_calculate_metrics_returns_three_values(self):
        """Verify the function returns exactly 3 values."""
        from qualtran.surface_code.flasq.error_mitigation import calculate_error_mitigation_metrics

        summary = self._mock_summary_for_metrics(
            total_spacetime_volume=100.0,
            total_t_count=10.0,
            cultivation_volume=5.0,
            total_depth=10.0,
        )

        result = calculate_error_mitigation_metrics(
            flasq_summary=summary,
            time_per_surface_code_cycle=1e-6,
            code_distance=5,
            lambda_val=50.0,
            cultivation_error_rate=1e-6,
        )
        assert len(result) == 3

