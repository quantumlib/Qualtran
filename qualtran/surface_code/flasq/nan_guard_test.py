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

import sympy
import pytest
from frozendict import frozendict
from qualtran.surface_code.flasq.flasq_model import FLASQCostModel, apply_flasq_cost_model
from qualtran.surface_code.flasq.volume_counting import FLASQGateCounts
from qualtran.surface_code.flasq.span_counting import GateSpan
from qualtran.surface_code.flasq.measurement_depth import MeasurementDepth
from qualtran.surface_code.flasq.symbols import ROTATION_ERROR, V_CULT_FACTOR, T_REACT
from qualtran.surface_code.flasq.utils import substitute_until_fixed_point
from qualtran.surface_code.flasq.optimization import generate_circuit_specific_configs
from qualtran.surface_code.flasq.examples.hwp import build_parallel_rz_circuit

def test_apply_flasq_cost_model_zero_fluid_ancilla():
    """Verify apply_flasq_cost_model handles n_fluid_ancilla=0 without producing zoo."""
    counts = FLASQGateCounts(
        t=0, toffoli=0, and_gate=0, and_dagger_gate=0,
        hadamard=10, s_gate=5, cnot=20, cz=0,
        z_rotation=100, x_rotation=0,
    )
    span = GateSpan(connect_span=50, compute_span=30)
    meas_depth = MeasurementDepth(depth=200)

    summary = apply_flasq_cost_model(
        model=FLASQCostModel(),
        n_total_logical_qubits=10,  # Same as qubit_counts → n_fluid_ancilla=0
        qubit_counts=10,
        counts=counts,
        span_info=span,
        measurement_depth=meas_depth,
        logical_timesteps_per_measurement=0,
    )

    assert summary.n_fluid_ancilla == 0
    assert summary.volume_limited_depth == sympy.oo
    # The summary should not contain zoo anywhere
    assert sympy.zoo not in summary.total_depth.atoms()

def test_resolve_symbols_with_zero_fluid_ancilla():
    """Verify resolve_symbols doesn't crash on a summary with n_fluid_ancilla=0."""
    counts = FLASQGateCounts(
        t=0, toffoli=0, and_gate=0, and_dagger_gate=0,
        hadamard=10, s_gate=5, cnot=20, cz=0,
        z_rotation=100, x_rotation=0,
    )
    span = GateSpan(connect_span=50, compute_span=30)
    meas_depth = MeasurementDepth(depth=200)

    summary = apply_flasq_cost_model(
        model=FLASQCostModel(),
        n_total_logical_qubits=10,
        qubit_counts=10,
        counts=counts,
        span_info=span,
        measurement_depth=meas_depth,
        logical_timesteps_per_measurement=0,
    )

    # Partial resolution (only ROTATION_ERROR) should not crash
    resolved = summary.resolve_symbols(
        frozendict({ROTATION_ERROR: 1e-7})
    )
    # total_t_count should be a number
    assert isinstance(resolved.total_t_count, (int, float))

    # Full resolution should also work
    resolved_full = summary.resolve_symbols(
        frozendict({ROTATION_ERROR: 1e-7, V_CULT_FACTOR: 1.0, T_REACT: 1.0})
    )
    assert isinstance(resolved_full.total_t_count, (int, float))

def test_substitute_until_fixed_point_with_zoo():
    """Verify substitute_until_fixed_point doesn't crash on zoo-containing expressions."""
    x = sympy.Symbol('x')
    expr = sympy.Max(0, sympy.zoo * x)

    # Should not raise ValueError
    result = substitute_until_fixed_point(
        expr, frozendict({x: 1.0}), try_make_number=True
    )
    # We don't care about the exact result — just that it doesn't crash
    assert result is not None

@pytest.mark.slow
def test_generate_circuit_specific_configs_does_not_crash():
    """Integration test: generate_circuit_specific_configs should work end-to-end."""
    build_rz = lambda **kwargs: build_parallel_rz_circuit(**kwargs)[0]
    kwargs = frozendict({"n_qubits_data": 7, "angle": 0.123})

    core_config, total_rot_err = generate_circuit_specific_configs(
        circuit_builder=build_rz,
        circuit_builder_kwargs=kwargs,
        total_synthesis_error=7e-7,
        total_cultivation_error=7e-5,
        phys_error_rate=1e-3,
        reference_code_distance=14,
    )

    assert core_config.code_distance == 14
    assert total_rot_err == 7e-7
    assert core_config.vcult_factor > 0
