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

from typing import Any, Dict, Tuple, Union
from unittest.mock import MagicMock, patch

import cirq
import numpy as np
import pandas as pd
import pytest
import sympy

# Define a simple frozen Bloq for testing unknown cases (should be hashable)
# This mimics how CirqGateAsBloq or other custom Bloqs might appear in unknown lists
from attrs import frozen
from frozendict import frozendict
from sympy import Symbol

from qualtran import Bloq, Signature
from qualtran.bloqs.basic_gates import CNOT, Hadamard, Ry, TGate, Toffoli, ZPowGate
from qualtran.surface_code.flasq import cultivation_analysis  # For the new test
from qualtran.surface_code.flasq.examples.ising import build_ising_circuit  # For integration test
from qualtran.surface_code.flasq.flasq_model import (
    apply_flasq_cost_model,
    conservative_FLASQ_costs,
    FLASQSummary,
    optimistic_FLASQ_costs,
    ROTATION_ERROR,
    V_CULT_FACTOR,
)
from qualtran.surface_code.flasq.measurement_depth import MeasurementDepth, TotalMeasurementDepth
from qualtran.surface_code.flasq.optimization import (
    analyze_logical_circuit,
    calculate_single_flasq_summary,
    CoreParametersConfig,
    ErrorBudget,
    generate_configs_for_constrained_qec,
    generate_configs_for_specific_cultivation_assumptions,
    generate_configs_from_cultivation_data,
    post_process_for_failure_budget,
    post_process_for_logical_depth,
    post_process_for_pec_runtime,
    run_sweep,
    SweepResult,
)
from qualtran.surface_code.flasq.span_counting import (
    BloqWithSpanInfo,
    GateSpan,
    TotalSpanCost,
)
from qualtran.surface_code.flasq.symbols import (
    T_REACT,
)
from qualtran.surface_code.flasq.volume_counting import (
    FLASQGateCounts,
    FLASQGateTotals,
)


@frozen
class _HashableUnknownBloq(Bloq):
    name: str

    @property
    def signature(self) -> Signature:
        return Signature.build(q=1)


# A simple circuit builder for testing
def _simple_circuit_builder(num_qubits: int, add_rotation: bool):
    q = cirq.LineQubit.range(num_qubits)
    circuit = cirq.Circuit(cirq.H(q[0]))
    if add_rotation and num_qubits > 0:
        circuit.append(cirq.Rz(rads=0.123).on(q[0]))
    if num_qubits > 1:
        circuit.append(cirq.CNOT(q[0], q[1]))
    return circuit


# A builder that returns a tuple of (circuit, kwargs)
def _tuple_returning_circuit_builder(
    num_qubits: int, add_rotation: bool
) -> Tuple[cirq.Circuit, Dict[str, Any]]:
    circuit = _simple_circuit_builder(num_qubits, add_rotation)
    conversion_kwargs = {"some_arg": "some_value"}
    return circuit, conversion_kwargs


def test_flasq_gate_counts_hashable():
    """Tests hashability of FLASQGateCounts, including with unknown bloqs."""
    # Case 1: Basic counts
    counts1 = FLASQGateCounts(t=10, cnot=5)
    assert hash(counts1) is not None

    # Case 2: With unknown bloqs (using standard hashable Bloqs)
    unknown_bloqs1: Dict[Bloq, sympy.Symbol] = {CNOT(): 2, Toffoli(): 1}
    counts2 = FLASQGateCounts(t=10, bloqs_with_unknown_cost=unknown_bloqs1)
    assert hash(counts2) is not None

    # Case 3: With unknown bloqs (using custom hashable Bloqs)
    unknown_bloqs2: Dict[Bloq, int] = {
        _HashableUnknownBloq("custom1"): 3,
        _HashableUnknownBloq("custom2"): 1,
    }
    counts3 = FLASQGateCounts(cnot=7, bloqs_with_unknown_cost=unknown_bloqs2)
    assert hash(counts3) is not None

    # Case 4: Equality implies same hash
    counts4 = FLASQGateCounts(t=10, cnot=5)
    assert counts1 == counts4
    assert hash(counts1) == hash(counts4)

    unknown_bloqs3: Dict[Bloq, sympy.Symbol] = {
        CNOT(): 2,
        Toffoli(): 1,
    }  # Same as unknown_bloqs1
    counts5 = FLASQGateCounts(t=10, bloqs_with_unknown_cost=unknown_bloqs3)
    assert counts2 == counts5
    assert hash(counts2) == hash(counts5)


def test_gate_span_hashable():
    """Tests hashability of GateSpan, including with uncounted bloqs."""
    # Case 1: Basic span
    span1 = GateSpan(connect_span=100, compute_span=200)
    assert hash(span1) is not None

    # Case 2: With uncounted bloqs (using standard hashable Bloqs)
    uncounted_bloqs1: Dict[Bloq, sympy.Symbol] = {Hadamard(): 10, CNOT(): 5}
    span2 = GateSpan(
        connect_span=100, compute_span=200, uncounted_bloqs=uncounted_bloqs1
    )
    assert hash(span2) is not None

    # Case 3: With uncounted bloqs (using custom hashable Bloqs)
    uncounted_bloqs2: Dict[Bloq, int] = {
        _HashableUnknownBloq("span_unknown1"): 2,
        _HashableUnknownBloq("span_unknown2"): 4,
    }
    span3 = GateSpan(
        connect_span=50, compute_span=100, uncounted_bloqs=uncounted_bloqs2
    )
    assert hash(span3) is not None

    # Case 4: Equality implies same hash
    span4 = GateSpan(connect_span=100, compute_span=200)
    assert span1 == span4
    assert hash(span1) == hash(span4)

    uncounted_bloqs3: Dict[Bloq, sympy.Symbol] = {
        Hadamard(): 10,
        CNOT(): 5,
    }  # Same as uncounted_bloqs1
    span5 = GateSpan(
        connect_span=100, compute_span=200, uncounted_bloqs=uncounted_bloqs3
    )
    assert span2 == span5
    assert hash(span2) == hash(span5)

    # Test with BloqWithSpanInfo (which is also frozen)
    wrapped_cnot = BloqWithSpanInfo(wrapped_bloq=CNOT(), connect_span=3, compute_span=3)
    span6 = GateSpan(uncounted_bloqs={wrapped_cnot: 1})
    assert hash(span6) is not None


def test_measurement_depth_hashable():
    """Tests hashability of MeasurementDepth, including with unknown bloqs."""
    # Case 1: Basic depth
    depth1 = MeasurementDepth(depth=10.5)
    assert hash(depth1) is not None

    # Case 2: With unknown bloqs (using standard hashable Bloqs)
    unknown_bloqs1: Dict[Bloq, sympy.Symbol] = {
        Ry(Symbol("theta")): 1,
        ZPowGate(exponent=0.1): 2,
    }
    depth2 = MeasurementDepth(depth=10.5, bloqs_with_unknown_depth=unknown_bloqs1)
    assert hash(depth2) is not None

    # Case 3: With unknown bloqs (using custom hashable Bloqs)
    unknown_bloqs2: Dict[Bloq, int] = {
        _HashableUnknownBloq("depth_unknown1"): 5,
        _HashableUnknownBloq("depth_unknown2"): 3,
    }
    depth3 = MeasurementDepth(depth=20, bloqs_with_unknown_depth=unknown_bloqs2)
    assert hash(depth3) is not None

    # Case 4: Equality implies same hash
    depth4 = MeasurementDepth(depth=10.5)
    assert depth1 == depth4
    assert hash(depth1) == hash(depth4)

    unknown_bloqs3: Dict[Bloq, sympy.Symbol] = {
        Ry(Symbol("theta")): 1,
        ZPowGate(exponent=0.1): 2,
    }  # Same as unknown_bloqs1
    depth5 = MeasurementDepth(depth=10.5, bloqs_with_unknown_depth=unknown_bloqs3)
    assert depth2 == depth5
    assert hash(depth2) == hash(depth5)


def test_total_measurement_depth_cost_key_hashable():
    """Tests hashability of TotalMeasurementDepth CostKey."""
    # Case 1: Default
    cost_key1 = TotalMeasurementDepth()
    assert hash(cost_key1) is not None

    # Case 2: With rotation_depth
    cost_key2 = TotalMeasurementDepth(rotation_depth=4.5)
    assert hash(cost_key2) is not None

    # Case 3: Equality implies same hash
    cost_key3 = TotalMeasurementDepth(rotation_depth=4.5)
    assert cost_key2 == cost_key3
    assert hash(cost_key2) == hash(cost_key3)


def test_flasq_summary_hashable():
    """Tests hashability of FLASQSummary, including with symbolic values."""
    N = sympy.Symbol("N")
    M = sympy.Symbol("M")

    # Case 1: Concrete values
    summary1 = FLASQSummary(
        total_clifford_volume=1000.0,
        total_depth=20.0,
        n_algorithmic_qubits=100,
        n_fluid_ancilla=50,
        total_t_count=5000.0,
        total_rotation_count=0.0,
        measurement_depth_val=10.0,
        scaled_measurement_depth=10.0,
        volume_limited_depth=5.0,
        total_computational_volume=250.0,
        idling_volume=2000.0,
        clifford_computational_volume=1000.0,
        non_clifford_lattice_surgery_volume=150.0,
        cultivation_volume=170.0,
        total_spacetime_volume=3170.0,
    )
    assert hash(summary1) is not None

    # Case 2: With symbolic values
    summary2 = FLASQSummary(
        total_clifford_volume=N * 10 + M,
        total_depth=N / 5.0,
        n_algorithmic_qubits=N,
        n_fluid_ancilla=M + 1,
        total_t_count=M * 2.0,
        total_rotation_count=M,
        measurement_depth_val=N / 10.0,
        scaled_measurement_depth=N / 10.0,
        volume_limited_depth=(N * 8 + M * 4) / (M + 1.0),
        total_computational_volume=N * 8 + M * 4,
        idling_volume=N * (N / 5),
        clifford_computational_volume=N * 8,
        non_clifford_lattice_surgery_volume=M * 3,
        cultivation_volume=M * 4 + V_CULT_FACTOR,
        total_spacetime_volume=N * 8 + M * 4 + V_CULT_FACTOR + N * (N / 5),
    )
    assert hash(summary2) is not None

    # Case 3: Equality implies same hash
    summary3 = FLASQSummary(
        total_clifford_volume=1000.0,
        total_depth=20.0,
        n_algorithmic_qubits=100,
        n_fluid_ancilla=50,
        total_t_count=5000.0,
        total_rotation_count=0.0,
        measurement_depth_val=10.0,
        scaled_measurement_depth=10.0,
        volume_limited_depth=5.0,
        total_computational_volume=250.0,
        idling_volume=2000.0,
        clifford_computational_volume=1000.0,
        non_clifford_lattice_surgery_volume=150.0,
        cultivation_volume=170.0,
        total_spacetime_volume=3170.0,
    )
    assert summary1 == summary3
    assert hash(summary1) == hash(summary3)

    summary4 = FLASQSummary(  # Same as summary2
        total_clifford_volume=N * 10 + M,
        total_depth=N / 5.0,
        n_algorithmic_qubits=N,
        n_fluid_ancilla=M + 1,
        total_t_count=M * 2.0,
        total_rotation_count=M,
        measurement_depth_val=N / 10.0,
        scaled_measurement_depth=N / 10.0,
        volume_limited_depth=(N * 8 + M * 4) / (M + 1.0),
        total_computational_volume=N * 8 + M * 4,
        idling_volume=N * (N / 5),
        clifford_computational_volume=N * 8,
        non_clifford_lattice_surgery_volume=M * 3,
        cultivation_volume=M * 4 + V_CULT_FACTOR,
        total_spacetime_volume=N * 8 + M * 4 + V_CULT_FACTOR + N * (N / 5),
    )
    assert summary2 == summary4
    assert hash(summary2) == hash(summary4)


def test_flasq_summary_resolved_hashable():
    """Tests hashability of FLASQSummary after resolving symbols."""
    N = sympy.Symbol("N")
    M = sympy.Symbol("M")

    symbolic_summary = FLASQSummary(
        total_clifford_volume=N * 10 + M,
        total_depth=N / 5.0,
        n_algorithmic_qubits=N,
        n_fluid_ancilla=M + 1,
        total_t_count=M * 2.0,
        total_rotation_count=M,
        measurement_depth_val=N / 10.0,
        scaled_measurement_depth=N / 10.0,
        volume_limited_depth=(N * 8 + M * 4) / (M + 1.0),
        total_computational_volume=N * 8 + M * 4,
        idling_volume=N * (N / 5),
        clifford_computational_volume=N * 8,
        non_clifford_lattice_surgery_volume=M * 3,
        cultivation_volume=M * 4 + V_CULT_FACTOR,
        total_spacetime_volume=N * 8 + M * 4 + V_CULT_FACTOR + N * (N / 5),
    )

    assumptions: Dict[Union[sympy.Symbol, str], Any] = {
        N: 100,
        M: 50,
        V_CULT_FACTOR: 6.0,
    }

    resolved_summary = symbolic_summary.resolve_symbols(frozendict(assumptions))

    # The resolved summary should now contain concrete numbers or simplified sympy expressions
    # It should still be hashable.
    assert hash(resolved_summary) is not None

    # Test equality and hash with another resolved summary
    symbolic_summary_copy = FLASQSummary(  # Same as symbolic_summary
        total_clifford_volume=N * 10 + M,
        total_depth=N / 5.0,
        n_algorithmic_qubits=N,
        n_fluid_ancilla=M + 1,
        total_t_count=M * 2.0,
        total_rotation_count=M,
        measurement_depth_val=N / 10.0,
        scaled_measurement_depth=N / 10.0,
        volume_limited_depth=(N * 8 + M * 4) / (M + 1.0),
        total_computational_volume=N * 8 + M * 4,
        idling_volume=N * (N / 5),
        clifford_computational_volume=N * 8,
        non_clifford_lattice_surgery_volume=M * 3,
        cultivation_volume=M * 4 + V_CULT_FACTOR,
        total_spacetime_volume=N * 8 + M * 4 + V_CULT_FACTOR + N * (N / 5),
    )
    resolved_summary_copy = symbolic_summary_copy.resolve_symbols(
        frozendict(assumptions)
    )

    assert resolved_summary == resolved_summary_copy
    assert hash(resolved_summary) == hash(resolved_summary_copy)


def test_frozendict_hashable():
    """Sanity check that frozendict is hashable."""
    d1 = frozendict({"a": 1, "b": 2})
    d2 = frozendict({"b": 2, "a": 1})  # Order shouldn't matter
    d3 = frozendict({"a": 1, "b": 3})

    assert hash(d1) is not None
    assert d1 == d2
    assert hash(d1) == hash(d2)
    assert d1 != d3
    # Hashes might collide, but for distinct objects they should ideally be different
    assert hash(d1) != hash(
        d3
    )  # This is probabilistic, but usually holds for simple cases

    # Test frozendict with Bloq keys
    bloq_dict1 = frozendict({CNOT(): 1, Hadamard(): 2})
    bloq_dict2 = frozendict({Hadamard(): 2, CNOT(): 1})
    bloq_dict3 = frozendict({CNOT(): 1, Hadamard(): 3})

    assert hash(bloq_dict1) is not None
    assert bloq_dict1 == bloq_dict2
    assert hash(bloq_dict1) == hash(bloq_dict2)
    assert bloq_dict1 != bloq_dict3
    assert hash(bloq_dict1) != hash(bloq_dict3)

    # Test frozendict with custom hashable Bloq keys
    custom_bloq_dict1 = frozendict(
        {_HashableUnknownBloq("A"): 1, _HashableUnknownBloq("B"): 2}
    )
    custom_bloq_dict2 = frozendict(
        {_HashableUnknownBloq("B"): 2, _HashableUnknownBloq("A"): 1}
    )
    custom_bloq_dict3 = frozendict(
        {_HashableUnknownBloq("A"): 1, _HashableUnknownBloq("B"): 3}
    )

    assert hash(custom_bloq_dict1) is not None
    assert custom_bloq_dict1 == custom_bloq_dict2
    assert hash(custom_bloq_dict1) == hash(custom_bloq_dict2)
    assert custom_bloq_dict1 != custom_bloq_dict3
    assert hash(custom_bloq_dict1) != hash(custom_bloq_dict3)  # Probabilistic


class OptimizationFunctionsTestSuite:
    def test_analyze_logical_circuit(self):
        kwargs = frozendict({"num_qubits": 2, "add_rotation": True})
        total_rot_error = 0.01

        res1 = analyze_logical_circuit(
            circuit_builder_func=_simple_circuit_builder,
            circuit_builder_kwargs=kwargs,
            total_allowable_rotation_error=total_rot_error,
        )

        assert isinstance(res1, frozendict)
        assert "flasq_counts" in res1
        assert isinstance(res1["flasq_counts"], FLASQGateCounts)
        assert res1["flasq_counts"].hadamard >= 1
        assert res1["flasq_counts"].cnot >= 1
        assert res1["flasq_counts"].z_rotation >= 1
        assert "total_span" in res1
        assert isinstance(res1["total_span"], GateSpan)
        assert "measurement_depth" in res1
        assert isinstance(res1["measurement_depth"], MeasurementDepth)
        assert "individual_allowable_rotation_error" in res1
        assert (
            res1["individual_allowable_rotation_error"]
            == total_rot_error / res1["flasq_counts"].total_rotations
        )
        assert "qubit_counts" in res1
        assert res1["qubit_counts"] == 2

        # Test caching
        res2 = analyze_logical_circuit(
            circuit_builder_func=_simple_circuit_builder,
            circuit_builder_kwargs=kwargs,  # Same kwargs
            total_allowable_rotation_error=total_rot_error,
        )
        assert res1 is res2  # Should be cached object
        assert analyze_logical_circuit.cache_info().hits >= 1

        # Test no rotations
        kwargs_no_rot = frozendict({"num_qubits": 1, "add_rotation": False})
        res_no_rot = analyze_logical_circuit(
            circuit_builder_func=_simple_circuit_builder,
            circuit_builder_kwargs=kwargs_no_rot,
            total_allowable_rotation_error=total_rot_error,
        )
        assert res_no_rot["flasq_counts"].total_rotations == 0
        assert (
            res_no_rot["individual_allowable_rotation_error"] == 1.0
        )  # Default for no rotations

        # Test zero rotation error with rotations (should raise ValueError)
        kwargs_with_rot = frozendict({"num_qubits": 1, "add_rotation": True})
        with pytest.raises(
            ValueError,
            match="total_allowable_rotation_error cannot be 0 if there are rotations",
        ):
            analyze_logical_circuit(
                circuit_builder_func=_simple_circuit_builder,
                circuit_builder_kwargs=kwargs_with_rot,
                total_allowable_rotation_error=0.0,
            )

    def test_analyze_logical_circuit_with_tuple_return(self):
        """Tests that analyze_logical_circuit correctly handles a builder
        that returns a (circuit, kwargs) tuple."""
        kwargs = frozendict({"num_qubits": 2, "add_rotation": True})
        total_rot_error = 0.01

        # Mock convert_circuit_for_flasq_analysis to check its arguments
        with patch(
            "qualtran.surface_code.flasq.optimization.analysis.convert_circuit_for_flasq_analysis"
        ) as mock_convert:
            # Set a valid return value for the mock
            from qualtran.bloqs.basic_gates import TGate

            mock_convert.return_value = (TGate(), cirq.Circuit(cirq.T(cirq.q(0))))

            res = analyze_logical_circuit(
                circuit_builder_func=_tuple_returning_circuit_builder,
                circuit_builder_kwargs=kwargs,
                total_allowable_rotation_error=total_rot_error,
            )

            # Check that the mock was called
            mock_convert.assert_called_once()
            # Check that the kwargs from the tuple were passed to the mock
            call_args, call_kwargs = mock_convert.call_args
            assert "some_arg" in call_kwargs
            assert call_kwargs["some_arg"] == "some_value"

    def test_calculate_single_flasq_summary(self):
        # Mock logical_circuit_analysis
        mock_logical_analysis = frozendict(
            {
                "flasq_counts": FLASQGateCounts(t=100, cnot=50, z_rotation=20),
                "total_span": GateSpan(connect_span=30, compute_span=30),
                "measurement_depth": MeasurementDepth(depth=15),
                "individual_allowable_rotation_error": 0.001,
                "qubit_counts": 10,
                "flasq_conversion_kwargs": frozendict({}),
            }
        )
        n_phys_qubits = 10000
        code_distance = 7
        lcpm = 10 / code_distance

        summary_conservative = calculate_single_flasq_summary(
            logical_circuit_analysis=mock_logical_analysis,
            n_phys_qubits=n_phys_qubits,
            code_distance=code_distance,
            flasq_model_obj=conservative_FLASQ_costs,
            logical_timesteps_per_measurement=lcpm,
        )
        assert isinstance(summary_conservative, FLASQSummary)
        assert summary_conservative.n_algorithmic_qubits == 10

        # Test caching
        summary_conservative_cached = calculate_single_flasq_summary(
            logical_circuit_analysis=mock_logical_analysis,  # Same
            n_phys_qubits=n_phys_qubits,
            code_distance=code_distance,
            flasq_model_obj=conservative_FLASQ_costs,  # Same
            logical_timesteps_per_measurement=lcpm,
        )
        assert summary_conservative is summary_conservative_cached
        assert calculate_single_flasq_summary.cache_info().hits >= 1

        # Test no fluid ancillas
        summary_none = calculate_single_flasq_summary(
            logical_circuit_analysis=mock_logical_analysis,
            n_phys_qubits=100,  # Too few physical qubits
            code_distance=code_distance,
            flasq_model_obj=conservative_FLASQ_costs,
            logical_timesteps_per_measurement=lcpm,
        )
        assert summary_none is None

    @pytest.mark.slow
    def test_ising_optimization_pipeline_integration(self):
        """Integration test for the full pipeline with a 4x4 Ising model."""
        # Fixed parameters for a single data point
        ising_params = frozendict(
            {
                "rows": 4,
                "cols": 4,
                "j_coupling": 1.0,
                "h_field": 3.0,
                "dt": 0.04,
                "n_steps": 2,
            }
        )
        total_rot_err = 0.005

        logical_analysis = analyze_logical_circuit(
            circuit_builder_func=build_ising_circuit,
            circuit_builder_kwargs=ising_params,
            total_allowable_rotation_error=total_rot_err,
        )
        assert isinstance(logical_analysis["flasq_counts"], FLASQGateCounts)
        assert (
            logical_analysis["qubit_counts"]
            == ising_params["rows"] * ising_params["cols"]
        )

        flasq_summary_conservative = calculate_single_flasq_summary(
            logical_circuit_analysis=logical_analysis,
            n_phys_qubits=50000,  # Ensure enough for fluid ancillas
            code_distance=13,
            flasq_model_obj=conservative_FLASQ_costs,
            logical_timesteps_per_measurement=(10 / 13),
        )
        assert isinstance(flasq_summary_conservative, FLASQSummary)

        resolved_flasq_summary = flasq_summary_conservative.resolve_symbols(
            frozendict(
                {
                    ROTATION_ERROR: logical_analysis[
                        "individual_allowable_rotation_error"
                    ],
                    V_CULT_FACTOR: 6.0,  # Added for completeness
                    T_REACT: 10.0 / 13,  # reaction_time_in_cycles / code_distance
                }
            )
        )

        # This part of the test now implicitly tests `post_process_for_pec_runtime`
        # by creating a dummy SweepResult and processing it.
        sweep_result = SweepResult(
            circuit_builder_kwargs=ising_params,
            core_config=CoreParametersConfig(
                code_distance=13,
                phys_error_rate=1e-3,
                cultivation_error_rate=1e-8,
                vcult_factor=6.0,
            ),
            total_allowable_rotation_error=total_rot_err,
            reaction_time_in_cycles=10.0,
            flasq_model_config=(conservative_FLASQ_costs, "Conservative"),
            n_phys_qubits=50000,
            logical_circuit_analysis=logical_analysis,
            # flasq_conversion_kwargs is part of logical_analysis
            flasq_summary=flasq_summary_conservative,
        )
        df = post_process_for_pec_runtime(
            [sweep_result], time_per_surface_code_cycle=1e-6
        )
        final_data = df.iloc[0]

        assert "Effective Time per Sample (s)" in final_data
        assert final_data["FLASQ Model"] == "Conservative"
        assert final_data["circuit_arg_rows"] == 4
        assert (
            final_data["Number of Algorithmic Qubits"]
            == ising_params["rows"] * ising_params["cols"]
        )
        assert final_data["Code Distance"] == 13


@pytest.mark.slow
def test_sweep_with_cultivation_data_derived_params():
    """
    Integration test using cultivation data to derive cultivation parameters
    for the optimization sweep.
    """
    # 1. Fixed Parameters for the sweep
    ising_circuit_kwargs = frozendict(
        {
            "rows": 6,
            "cols": 6,
            "j_coupling": 1.0,
            "h_field": 3.04438,
            "dt": 0.04,
            "n_steps": 2,
        }
    )
    code_distance_val = 15  # Single code distance for this test

    # 2. Use a physical error rate relevant to the sweep for fetching cultivation data
    phys_error_rate_for_cult_data = 0.001

    # 3. Generate CoreParametersConfig list using the helper
    core_configs = generate_configs_from_cultivation_data(
        code_distance_list=[code_distance_val],
        phys_error_rate_list=[phys_error_rate_for_cult_data],
        cultivation_data_source_distance_list=[3, 5],  # Test both sources
        cultivation_data_sampling_frequency=1000,  # Take the "best" (last) row
    )

    if not core_configs:  # pragma: no cover
        pytest.skip(
            f"No CoreParametersConfig objects were generated from cultivation data for d={code_distance_val}, p_err={phys_error_rate_for_cult_data}."
        )

    # 4. Run the optimization sweep
    sweep_results = run_sweep(
        circuit_builder_func=build_ising_circuit,
        circuit_builder_kwargs_list=[ising_circuit_kwargs],
        core_configs_list=core_configs,
        total_allowable_rotation_error_list=0.01,
        n_phys_qubits_total_list=[30000],
        flasq_model_configs=[(conservative_FLASQ_costs, "Conservative")],
        reaction_time_in_cycles_list=10.0,
        print_level=0,  # Suppress output for test
    )

    # 5. Assertions
    results_list = post_process_for_pec_runtime(
        sweep_results, time_per_surface_code_cycle=1e-6
    ).to_dict("records")
    assert len(results_list) == len(core_configs), (
        "Number of results should match number of generated core_configs. "
        f"Got {len(results_list)} results, expected {len(core_configs)} core_configs."
    )

    for i, res in enumerate(results_list):
        assert res["circuit_arg_rows"] == ising_circuit_kwargs["rows"]
        assert res["Physical Error Rate"] == core_configs[i].phys_error_rate
        assert res["Code Distance"] == core_configs[i].code_distance
        assert np.isclose(
            res["Cultivation Error Rate"],
            core_configs[i].cultivation_error_rate,
        )
        assert np.isclose(res["V_CULT Factor"], core_configs[i].vcult_factor)
        assert "Effective Time per Sample (s)" in res
        assert (
            res["circuit_arg_cultivation_data_source_distance"]
            == core_configs[i].cultivation_data_source_distance
        )


class HelperFunctionsForSweepTestSuite:
    def test_generate_configs_for_specific_cultivation_assumptions(self):
        configs = generate_configs_for_specific_cultivation_assumptions(
            code_distance_list=[7, 9],
            phys_error_rate_list=[1e-3, 1e-4],
            cultivation_error_rate=1e-8,
            vcult_factor=6.0,
        )
        assert len(configs) == 4  # 2 distances * 2 phys_error_rates
        for cfg in configs:
            assert isinstance(cfg, CoreParametersConfig)
            assert cfg.cultivation_error_rate == 1e-8
            assert cfg.vcult_factor == 6.0
            assert cfg.cultivation_data_source_distance is None
            assert cfg.code_distance in [7, 9]
            assert cfg.phys_error_rate in [1e-3, 1e-4]

    def test_generate_configs_from_cultivation_data(self, monkeypatch):
        # Mock get_regularized_filtered_cultivation_data
        mock_cult_data_dist3 = pd.DataFrame(
            {
                "t_gate_cultivation_error_rate": [5e-9, 4e-9],
                "expected_volume": [1000.0, 1200.0],
                "cultivation_distance": [3, 3],  # For verification
            },
        )
        mock_cult_data_dist5 = pd.DataFrame(
            {
                "t_gate_cultivation_error_rate": [3e-9],
                "expected_volume": [1500.0],
                "cultivation_distance": [5],  # Corrected: should be int
            }
        )
        mock_cult_data_empty = pd.DataFrame(
            columns=["t_gate_cultivation_error_rate", "expected_volume"]
        )

        def mock_get_cult_data(error_rate, cultivation_distance, **kwargs):
            if cultivation_distance == 3 and np.isclose(error_rate, 1e-3):
                return mock_cult_data_dist3
            elif cultivation_distance == 5 and np.isclose(error_rate, 1e-3):
                return mock_cult_data_dist5
            return mock_cult_data_empty

        monkeypatch.setattr(
            cultivation_analysis,
            "get_regularized_filtered_cultivation_data",
            mock_get_cult_data,
        )

        # Test case 1: cultivation_data_sampling_frequency = None (take all)
        configs_none = generate_configs_from_cultivation_data(
            code_distance_list=11,
            phys_error_rate_list=1e-3,
            cultivation_data_source_distance_list=[3, 5],  # Test both
            cultivation_data_sampling_frequency=None,  # Use all rows from mock
        )
        # Expected: 2 from dist3_data + 1 from dist5_data = 3 configs
        assert len(configs_none) == 3

        d_val = 11
        phys_err_val = 1e-3

        # Check first config (from dist3, first row)
        # Note: iterrows() goes in order of DataFrame, sampling logic doesn't reorder original mock
        cfg1 = configs_none[0]
        assert cfg1.code_distance == d_val
        assert cfg1.phys_error_rate == phys_err_val
        assert cfg1.cultivation_error_rate == 5e-9
        assert np.isclose(cfg1.vcult_factor, 1000.0 / (2 * (d_val + 1) ** 2 * d_val))
        assert cfg1.cultivation_data_source_distance == 3

        # Check second config (from dist3, second row of mock_cult_data_dist3)
        cfg2 = configs_none[1]
        assert cfg2.code_distance == d_val
        assert cfg2.phys_error_rate == phys_err_val
        assert cfg2.cultivation_error_rate == 4e-9
        assert np.isclose(cfg2.vcult_factor, 1200.0 / (2 * (d_val + 1) ** 2 * d_val))
        assert cfg2.cultivation_data_source_distance == 3

        # Check third config (from dist5, only row of mock_cult_data_dist5)
        cfg3 = configs_none[2]
        assert cfg3.code_distance == d_val
        assert cfg3.phys_error_rate == phys_err_val
        assert cfg3.cultivation_error_rate == 3e-9
        assert np.isclose(cfg3.vcult_factor, 1500.0 / (2 * (d_val + 1) ** 2 * d_val))
        assert cfg3.cultivation_data_source_distance == 5

        # Test case 2: cultivation_data_sampling_frequency = 1 (take all, sampled from tail)
        configs_k1 = generate_configs_from_cultivation_data(
            code_distance_list=11,
            phys_error_rate_list=1e-3,
            cultivation_data_source_distance_list=[3, 5],
            cultivation_data_sampling_frequency=1,
        )
        assert len(configs_k1) == 3
        # Order for dist3 will be reversed compared to DataFrame order due to tail sampling
        # cfg_k1_0 is from dist3, last row of mock_cult_data_dist3 (4e-9)
        # cfg_k1_1 is from dist3, first row of mock_cult_data_dist3 (5e-9)
        # cfg_k1_2 is from dist5 (3e-9)
        assert configs_k1[0].cultivation_error_rate == 4e-9  # Last row of dist3 data
        assert configs_k1[1].cultivation_error_rate == 5e-9  # First row of dist3 data
        assert configs_k1[2].cultivation_error_rate == 3e-9  # Only row of dist5 data

        # Test case 3: cultivation_data_sampling_frequency = 2
        # For dist3 (2 rows): takes last row. For dist5 (1 row): takes the only row.
        configs_k2 = generate_configs_from_cultivation_data(
            code_distance_list=11,
            phys_error_rate_list=1e-3,
            cultivation_data_source_distance_list=[3, 5],
            cultivation_data_sampling_frequency=2,
        )
        assert len(configs_k2) == 2  # 1 from dist3, 1 from dist5
        assert configs_k2[0].cultivation_error_rate == 4e-9  # Last row of dist3 data
        assert configs_k2[0].cultivation_data_source_distance == 3
        assert configs_k2[1].cultivation_error_rate == 3e-9  # Only row of dist5 data
        assert configs_k2[1].cultivation_data_source_distance == 5

        # Test case 4: cultivation_data_sampling_frequency = 100 (large number)
        # For dist3 (2 rows): takes last row. For dist5 (1 row): takes the only row.
        configs_k100 = generate_configs_from_cultivation_data(
            code_distance_list=11,
            phys_error_rate_list=1e-3,
            cultivation_data_source_distance_list=[3, 5],
            cultivation_data_sampling_frequency=100,
        )
        assert len(configs_k100) == 2  # 1 from dist3, 1 from dist5
        assert configs_k100[0].cultivation_error_rate == 4e-9  # Last row of dist3 data
        assert configs_k100[1].cultivation_error_rate == 3e-9  # Only row of dist5 data

        # Test case 5: Invalid cultivation_data_sampling_frequency
        with pytest.raises(
            ValueError,
            match="cultivation_data_sampling_frequency must be None or a positive integer.",
        ):
            generate_configs_from_cultivation_data(
                code_distance_list=11,
                phys_error_rate_list=1e-3,
                cultivation_data_source_distance_list=[3],
                cultivation_data_sampling_frequency=0,
            )


class NewSweepPipelineTestSuite:
    def test_run_sweep_single_point(self):
        """Test the new `run_sweep` function with a single parameter set."""
        circuit_kwargs = frozendict({"num_qubits": 2, "add_rotation": True})
        core_config = CoreParametersConfig(
            code_distance=7,
            phys_error_rate=1e-3,
            cultivation_error_rate=1e-8,
            vcult_factor=6.0,
        )
        results = run_sweep(
            circuit_builder_func=_simple_circuit_builder,
            circuit_builder_kwargs_list=[circuit_kwargs],
            core_configs_list=[core_config],
            total_allowable_rotation_error_list=[0.01],
            reaction_time_in_cycles_list=[10.0],
            flasq_model_configs=[(conservative_FLASQ_costs, "Conservative")],
            n_phys_qubits_total_list=[20000],
            print_level=0,
        )
        assert len(results) == 1
        res = results[0]
        assert isinstance(res, SweepResult)
        assert isinstance(res.flasq_summary, FLASQSummary)
        assert res.core_config.code_distance == 7
        assert res.core_config.phys_error_rate == 1e-3
        assert res.flasq_model_config[1] == "Conservative"
        assert res.circuit_builder_kwargs["num_qubits"] == 2

    def test_post_process_for_pec_runtime(self):
        """Test the `post_process_for_pec_runtime` function."""
        # Create a sample raw result from run_sweep
        logical_analysis = analyze_logical_circuit(
            circuit_builder_func=_simple_circuit_builder,
            circuit_builder_kwargs=frozendict({"num_qubits": 2, "add_rotation": True}),
            total_allowable_rotation_error=0.01,
        )
        flasq_summary = calculate_single_flasq_summary(
            logical_circuit_analysis=logical_analysis,
            n_phys_qubits=20000,
            code_distance=7,
            flasq_model_obj=conservative_FLASQ_costs,
            logical_timesteps_per_measurement=10.0 / 7,
        )
        core_config = CoreParametersConfig(
            code_distance=7,
            phys_error_rate=1e-3,
            cultivation_error_rate=1e-8,
            vcult_factor=6.0,
            cultivation_data_source_distance=None,
        )
        raw_result = SweepResult(
            circuit_builder_kwargs=frozendict({"num_qubits": 2, "add_rotation": True}),
            core_config=core_config,
            total_allowable_rotation_error=0.01,
            reaction_time_in_cycles=10.0,
            flasq_model_config=(conservative_FLASQ_costs, "Conservative"),
            n_phys_qubits=20000,
            logical_circuit_analysis=logical_analysis,
            flasq_summary=flasq_summary,
        )

        df = post_process_for_pec_runtime(
            sweep_results=[raw_result], time_per_surface_code_cycle=1e-6
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        row = df.iloc[0]
        assert "Effective Time per Sample (s)" in row
        assert row["Code Distance"] == 7
        assert row["FLASQ Model"] == "Conservative"
        assert row["circuit_arg_num_qubits"] == 2

    @pytest.mark.slow
    def test_full_new_pipeline(self):
        """Test `run_sweep` followed by `post_process_for_pec_runtime`."""
        results = run_sweep(
            circuit_builder_func=_simple_circuit_builder,
            circuit_builder_kwargs_list=frozendict(
                {"num_qubits": 2, "add_rotation": True}
            ),
            core_configs_list=CoreParametersConfig(
                code_distance=7,
                phys_error_rate=1e-3,
                cultivation_error_rate=1e-8,
                vcult_factor=6.0,
            ),
            total_allowable_rotation_error_list=0.01,
            reaction_time_in_cycles_list=10.0,
            flasq_model_configs=[(conservative_FLASQ_costs, "Conservative")],
            n_phys_qubits_total_list=[20000, 30000],
            print_level=0,
        )
        assert len(results) == 2
        df = post_process_for_pec_runtime(results, time_per_surface_code_cycle=1e-6)
        assert len(df) == 2
        assert "Wall Clock Time per Sample (s)" in df.columns
        assert all(df["Code Distance"] == 7)


class ConstrainedQECOptimizationTestSuite:
    # --- Tests for ErrorBudget ---
    def test_error_budget_class(self):
        budget = ErrorBudget(logical=0.001, cultivation=0.002, synthesis=0.003)
        assert budget.logical == 0.001
        assert budget.total == pytest.approx(0.006)

    # --- Tests for generate_configs_for_constrained_qec ---

    @patch("qualtran.surface_code.flasq.optimization.analysis.analyze_logical_circuit")
    @patch(
        "qualtran.surface_code.flasq.optimization.cultivation_analysis.find_best_cultivation_parameters"
    )
    @patch("qualtran.surface_code.flasq.optimization.cultivation_analysis.round_error_rate_up")
    def test_generate_configs_constrained_qec_feasible(
        self, mock_round_up, mock_find_best, mock_analyze
    ):
        # Setup: M=1000 (T=1000, No rotations), Budget(cultivation=0.1) -> required_p_mag=1e-4

        # Mock analyze_logical_circuit
        M = 1000
        mock_counts = FLASQGateCounts(t=M, z_rotation=0)
        mock_analyze.return_value = frozendict(
            {
                "flasq_counts": mock_counts,
                "individual_allowable_rotation_error": 1.0,  # No rotations
            }
        )

        # Mock cultivation data lookup
        p_phys = 1e-3
        mock_round_up.return_value = p_phys  # Assume rounding is successful

        # Mock find_best_cultivation_parameters to return a feasible result
        mock_best_params = pd.Series(
            {
                "t_gate_cultivation_error_rate": 9e-5,
                "expected_volume": 30000.0,  # Dummy volume
                "cultivation_distance": 5,
            }
        )
        mock_find_best.return_value = mock_best_params

        budget = ErrorBudget(logical=0.1, cultivation=0.1, synthesis=0.01)

        # Action
        configs = generate_configs_for_constrained_qec(
            circuit_builder_func=_simple_circuit_builder,  # Dummy func from the file
            circuit_builder_kwargs=frozendict({"num_qubits": 2, "add_rotation": False}),
            error_budget=budget,
            phys_error_rate_list=[p_phys],
            code_distance_list=[11, 13],
        )

        # Assertion
        assert len(configs) == 2

        # Check mock calls
        mock_analyze.assert_called_once()
        assert (
            mock_analyze.call_args.kwargs["total_allowable_rotation_error"]
            == budget.synthesis
        )

        # Check cultivation optimization call (required_p_mag = 0.1 / 1000 = 1e-4)
        mock_find_best.assert_called_once_with(
            physical_error_rate=p_phys,
            target_logical_error_rate=1e-4,
            decimal_precision=8,
            uncertainty_cutoff=100,
        )

        # Check config content (e.g., for d=11)
        cfg11 = next(c for c in configs if c.code_distance == 11)
        assert cfg11.phys_error_rate == p_phys
        assert cfg11.cultivation_error_rate == 9e-5
        # vcult = 30000 / (2 * 12^2 * 11) = 30000 / 3168 ≈ 9.4696
        assert cfg11.vcult_factor == pytest.approx(9.46969696)
        assert cfg11.cultivation_data_source_distance == 5

    # --- Tests for post_process_for_failure_budget ---

    # Helper to create mock SweepResults for post-processing tests
    def _create_mock_sweep_result(
        self, name, synthesis_budget=0.01, total_depth=100.0, code_distance=10
    ):
        # This helper simplifies creating the complex SweepResult object for testing post-processing.
        # We use MagicMock for the internal objects as the post-processing primarily interacts with their methods/attributes.

        # Mock resolved FLASQSummary
        resolved_summary = MagicMock(spec=FLASQSummary)
        resolved_summary.total_depth = total_depth
        # Add other necessary attributes for data assembly
        resolved_summary.total_t_count = 1000
        resolved_summary.n_algorithmic_qubits = 10
        resolved_summary.n_fluid_ancilla = 5

        # Mock the initial SweepResult object
        result = MagicMock(spec=SweepResult)
        # Mock the resolve_symbols method to return the resolved summary
        result.reaction_time_in_cycles = 10.0  # Needed for T_REACT calculation

        result.flasq_summary.resolve_symbols.return_value = resolved_summary

        result.total_allowable_rotation_error = synthesis_budget
        result.flasq_model_config = (conservative_FLASQ_costs, name)

        # Mock CoreParametersConfig
        core_config = MagicMock(spec=CoreParametersConfig)
        core_config.code_distance = code_distance
        core_config.phys_error_rate = 1e-3
        core_config.cultivation_error_rate = 1e-8
        core_config.vcult_factor = 5.0
        core_config.cultivation_data_source_distance = None
        result.core_config = core_config
        result.core_config.code_distance = (
            code_distance  # Needed for T_REACT calculation
        )

        result.n_phys_qubits = 50000
        result.circuit_builder_kwargs = frozendict({"circuit": name})
        # Mock logical_circuit_analysis for symbol resolution inputs
        result.logical_circuit_analysis = frozendict(
            {"individual_allowable_rotation_error": synthesis_budget}
        )

        return result

    @patch("qualtran.surface_code.flasq.optimization.postprocessing.calculate_failure_probabilities")
    def test_post_process_failure_budget_filtering(self, mock_calc_failures):
        """Test Case 1: Filtering Logic (Pass/Fail)."""

        budget = ErrorBudget(logical=0.01, cultivation=0.01, synthesis=0.01)

        res_pass = self._create_mock_sweep_result("Pass")
        res_fail_cliff = self._create_mock_sweep_result("FailCliff")
        res_fail_t = self._create_mock_sweep_result("FailT")

        # Configure mock_calc_failures return values based on the input summary (which is unique per result)
        def mock_probabilities_side_effect(flasq_summary, **kwargs):
            if flasq_summary == res_pass.flasq_summary.resolve_symbols.return_value:
                return (0.005, 0.005)  # Pass (P_fail_Clifford, P_fail_T)
            if (
                flasq_summary
                == res_fail_cliff.flasq_summary.resolve_symbols.return_value
            ):
                return (0.02, 0.005)  # Fail Clifford (P_fail_Clifford > budget.logical)
            if flasq_summary == res_fail_t.flasq_summary.resolve_symbols.return_value:
                return (0.005, 0.02)  # Fail T (P_fail_T > budget.cultivation)
            return (0, 0)

        mock_calc_failures.side_effect = mock_probabilities_side_effect

        # Execute
        df = post_process_for_failure_budget(
            sweep_results=[res_pass, res_fail_cliff, res_fail_t],
            error_budget=budget,
            time_per_surface_code_cycle=1e-6,
        )

        # Assertions
        assert len(df) == 1
        assert (
            df.iloc[0]["FLASQ Model"] == "Pass"
        )  # Only the 'Pass' result should remain
        assert df.iloc[0]["P_fail_Clifford (P_log)"] == 0.005
        assert df.iloc[0]["P_fail_T (P_dis)"] == 0.005
        assert df.iloc[0]["Sum of Failure Probabilities (P_log + P_dis)"] == 0.01

    @patch("qualtran.surface_code.flasq.optimization.postprocessing.calculate_failure_probabilities")
    def test_post_process_failure_budget_time_calc(self, mock_calc_failures):
        """Test Case 2: Wall Clock Time Calculation."""

        budget = ErrorBudget(logical=0.1, cultivation=0.1, synthesis=0.1)

        # Setup mock result: L=5000, d=15
        result = self._create_mock_sweep_result(
            "TestTime", total_depth=5000.0, code_distance=15, synthesis_budget=0.1
        )

        # Mock probabilities to pass
        mock_calc_failures.return_value = (0.01, 0.01)

        # Execute
        t_cyc = 400e-9  # 400 ns
        df = post_process_for_failure_budget(
            sweep_results=[result],
            error_budget=budget,
            time_per_surface_code_cycle=t_cyc,
        )

        # Assertion: Time = t_cyc * d * L
        # 400e-9 * 15 * 5000 = 0.03 seconds
        assert len(df) == 1
        assert pytest.approx(df.iloc[0]["Wall Clock Time (s)"]) == 0.03

    @patch("qualtran.surface_code.flasq.optimization.postprocessing.calculate_failure_probabilities")
    def test_post_process_failure_budget_mismatch(self, mock_calc_failures):
        """Test Case 3: Synthesis Budget Mismatch."""

        budget = ErrorBudget(logical=0.01, cultivation=0.01, synthesis=0.01)

        # Setup mock result with mismatching synthesis budget
        result = self._create_mock_sweep_result(
            "Mismatch", synthesis_budget=0.05
        )  # Mismatch!

        # Execute
        df = post_process_for_failure_budget(
            sweep_results=[result],
            error_budget=budget,
        )

        # Assertion
        assert len(df) == 0
        mock_calc_failures.assert_not_called()
