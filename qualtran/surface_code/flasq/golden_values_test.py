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

# golden_values_test.py
# Phase 1 golden value regression tests for the three example applications.
#
# These tests record the end-to-end numerical outputs of the FLASQ pipeline
# for fixed, paper-inspired parameters. Any refactoring that changes these
# values is a SEMANTIC change requiring human review.
#
# Parameters are drawn from the FLASQ paper (Table 1, demo notebooks).

import math

import numpy as np
import pytest
from frozendict import frozendict

from qualtran.resource_counting import get_cost_value, QubitCount
from qualtran.surface_code.flasq.cirq_interop import convert_circuit_for_flasq_analysis
from qualtran.surface_code.flasq.examples.hwp import build_hwp_circuit
from qualtran.surface_code.flasq.examples.ising import build_ising_circuit
from qualtran.surface_code.flasq.flasq_model import (
    apply_flasq_cost_model,
    conservative_FLASQ_costs,
    get_rotation_depth,
    optimistic_FLASQ_costs,
)
from qualtran.surface_code.flasq.measurement_depth import MeasurementDepth, TotalMeasurementDepth
from qualtran.surface_code.flasq.naive_grid_qubit_manager import NaiveGridQubitManager
from qualtran.surface_code.flasq.span_counting import TotalSpanCost
from qualtran.surface_code.flasq.symbols import ROTATION_ERROR, T_REACT, V_CULT_FACTOR
from qualtran.surface_code.flasq.volume_counting import FLASQGateCounts, FLASQGateTotals

STANDARD_ASSUMPTIONS = frozendict({
    ROTATION_ERROR: 1e-3,
    V_CULT_FACTOR: 6.0,
    T_REACT: 1.0,
})


def _run_ising_pipeline(rows, cols, n_steps, order=2, n_fluid_ancilla=100, model=conservative_FLASQ_costs):
    """Helper: run full FLASQ pipeline for an Ising model and return resolved summary."""
    circuit = build_ising_circuit(
        rows=rows, cols=cols, j_coupling=1.0, h_field=3.04438, dt=0.04,
        n_steps=n_steps, order=order,
    )
    cbloq, _ = convert_circuit_for_flasq_analysis(circuit)

    flasq_counts = get_cost_value(cbloq, FLASQGateTotals())
    span_info = get_cost_value(cbloq, TotalSpanCost())
    qubit_counts = get_cost_value(cbloq, QubitCount())

    individual_rotation_error = (
        0.005 / flasq_counts.total_rotations
        if flasq_counts.total_rotations > 0
        else 0.005
    )
    rotation_depth_val = get_rotation_depth(rotation_error=individual_rotation_error)
    measurement_depth = get_cost_value(
        cbloq, TotalMeasurementDepth(rotation_depth=rotation_depth_val)
    )

    n_total_logical_qubits = qubit_counts + n_fluid_ancilla
    summary = apply_flasq_cost_model(
        model=model,
        n_total_logical_qubits=n_total_logical_qubits,
        qubit_counts=qubit_counts,
        counts=flasq_counts,
        span_info=span_info,
        measurement_depth=measurement_depth,
        logical_timesteps_per_measurement=1.0,
    )
    resolved = summary.resolve_symbols(STANDARD_ASSUMPTIONS)
    return flasq_counts, qubit_counts, resolved


def _run_hwp_pipeline(n_qubits_data, angle=0.123, n_fluid_ancilla=20):
    """Helper: run full FLASQ pipeline for HWP and return resolved summary."""
    # IMPORTANT: Must pass BOTH managers explicitly. build_hwp_circuit has
    # mutable default NaiveGridQubitManager instances that accumulate state
    # across calls, making qubit positions (and thus spans) non-deterministic.
    data_manager = NaiveGridQubitManager(max_cols=10, negative=False)
    ancilla_manager = NaiveGridQubitManager(max_cols=20, negative=True)
    hwp_bloq, hwp_circuit, hwp_data_qubits = build_hwp_circuit(
        n_qubits_data=n_qubits_data, angle=angle,
        data_qubit_manager=data_manager,
        ancilla_qubit_manager=ancilla_manager,
    )
    in_quregs = {"x": np.asarray(hwp_data_qubits)}
    cbloq, _ = convert_circuit_for_flasq_analysis(
        hwp_circuit, signature=hwp_bloq.signature,
        qubit_manager=ancilla_manager,
        in_quregs=in_quregs, out_quregs=in_quregs,
    )

    flasq_counts = get_cost_value(cbloq, FLASQGateTotals())
    span_info = get_cost_value(cbloq, TotalSpanCost())
    qubit_counts = get_cost_value(cbloq, QubitCount())

    dummy_measurement_depth = MeasurementDepth(depth=0)
    n_total_logical_qubits = qubit_counts + n_fluid_ancilla

    summary = apply_flasq_cost_model(
        model=conservative_FLASQ_costs,
        n_total_logical_qubits=n_total_logical_qubits,
        qubit_counts=qubit_counts,
        counts=flasq_counts,
        span_info=span_info,
        measurement_depth=dummy_measurement_depth,
        logical_timesteps_per_measurement=1.0,
    )
    resolved = summary.resolve_symbols(STANDARD_ASSUMPTIONS)
    return flasq_counts, qubit_counts, resolved


# =============================================================================
# Ising Golden Values — Paper Table 1 inspired parameters
# =============================================================================


@pytest.mark.slow
class IsingGoldenValuesTestSuite:
    """End-to-end regression for Ising model with conservative FLASQ costs."""

    def test_ising_11x11_2nd_order_gate_counts(self):
        """Verify raw gate counts for 11x11, 20 steps, 2nd order."""
        counts, qubits, _ = _run_ising_pipeline(11, 11, 20, order=2)
        assert qubits == 121
        assert counts.cnot == 9680
        assert counts.z_rotation == 4840
        assert counts.x_rotation == 2541

    def test_ising_11x11_2nd_order_summary(self):
        """Golden values for 11x11, 20 steps, 2nd order Trotterization."""
        _, _, resolved = _run_ising_pipeline(11, 11, 20, order=2)
        assert resolved.total_spacetime_volume == pytest.approx(3906336.31, rel=1e-6)
        assert resolved.total_clifford_volume == pytest.approx(2226763.32, rel=1e-6)
        assert resolved.total_depth == pytest.approx(17675.73, rel=1e-4)
        assert float(resolved.total_t_count) == pytest.approx(74857.11, rel=1e-6)
        assert float(resolved.cultivation_volume) == pytest.approx(673713.99, rel=1e-6)

    def test_ising_10x10_4th_order_gate_counts(self):
        """Verify raw gate counts for 10x10, 20 steps, 4th order."""
        counts, qubits, _ = _run_ising_pipeline(10, 10, 20, order=4)
        assert qubits == 100
        assert counts.cnot == 40000
        assert counts.z_rotation == 20000
        assert counts.x_rotation == 10100

    def test_ising_10x10_4th_order_summary(self):
        """Golden values for 10x10, 20 steps, 4th order Trotterization."""
        _, _, resolved = _run_ising_pipeline(10, 10, 20, order=4)
        assert resolved.total_spacetime_volume == pytest.approx(14418725.64, rel=1e-6)
        assert resolved.total_clifford_volume == pytest.approx(7569362.82, rel=1e-6)
        assert resolved.total_depth == pytest.approx(72093.63, rel=1e-4)
        assert float(resolved.total_t_count) == pytest.approx(305270.16, rel=1e-6)
        assert float(resolved.cultivation_volume) == pytest.approx(2747431.41, rel=1e-6)


@pytest.mark.slow
class IsingOptimisticGoldenValuesTestSuite:
    """End-to-end regression for Ising model with optimistic FLASQ costs."""

    def test_ising_11x11_2nd_order_optimistic_summary(self):
        """Golden values for 11x11, 20 steps, 2nd order with optimistic costs."""
        _, _, resolved = _run_ising_pipeline(11, 11, 20, order=2, model=optimistic_FLASQ_costs)
        assert resolved.total_spacetime_volume == pytest.approx(2010595.37, rel=1e-6)
        assert resolved.total_clifford_volume == pytest.approx(1136023.71, rel=1e-6)
        assert resolved.total_depth == pytest.approx(9097.72, rel=1e-4)
        assert float(resolved.total_t_count) == pytest.approx(74857.11, rel=1e-6)
        assert float(resolved.cultivation_volume) == pytest.approx(449142.66, rel=1e-6)


# =============================================================================
# HWP Golden Values — Multiple N values at paper's target error rates
# =============================================================================


@pytest.mark.slow
class HWPGoldenValuesTestSuite:
    """End-to-end regression for Hamming Weight Phasing."""

    @pytest.mark.parametrize(
        "n_qubits, expected_qubit_count, expected_and, expected_cnot, expected_z_rot",
        [
            (7, 12, 4, 46, 3),
            (15, 27, 11, 118, 4),
            (43, 85, 39, 398, 6),
        ],
    )
    def test_hwp_gate_counts(
        self, n_qubits, expected_qubit_count, expected_and, expected_cnot, expected_z_rot
    ):
        """Verify raw gate counts for HWP at multiple N values."""
        counts, qubits, _ = _run_hwp_pipeline(n_qubits)
        assert qubits == expected_qubit_count
        assert counts.and_gate == expected_and
        assert counts.and_dagger_gate == expected_and
        assert counts.cnot == expected_cnot
        assert counts.z_rotation == expected_z_rot

    @pytest.mark.parametrize(
        "n_qubits, expected_stv, expected_cliff_vol, expected_t_count",
        [
            (7, 2625.06, 1534.40, 46.425597),
            (15, 8970.45, 6938.24, 84.567463),
            (43, 68818.69, 63475.37, 216.851194),
        ],
    )
    def test_hwp_summary(self, n_qubits, expected_stv, expected_cliff_vol, expected_t_count):
        """Golden values for HWP summary at multiple N values."""
        _, _, resolved = _run_hwp_pipeline(n_qubits)
        assert resolved.total_spacetime_volume == pytest.approx(expected_stv, rel=1e-4)
        assert resolved.total_clifford_volume == pytest.approx(expected_cliff_vol, rel=1e-4)
        assert float(resolved.total_t_count) == pytest.approx(expected_t_count, rel=1e-6)
