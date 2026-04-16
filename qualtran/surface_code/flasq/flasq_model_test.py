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

import attrs
import cirq
import numpy as np
import pytest
import sympy
from frozendict import frozendict

# Imports needed for test data
from qualtran import BloqBuilder
from qualtran.bloqs.basic_gates import CNOT, Hadamard, Ry, Rz, SGate, Toffoli
from qualtran.bloqs.mcmt import And
from qualtran.resource_counting import get_cost_value, QubitCount
from qualtran.surface_code.flasq.cirq_interop import convert_circuit_for_flasq_analysis
from qualtran.surface_code.flasq.flasq_model import (
    conservative_FLASQ_costs,  # Import the new instance
)
from qualtran.surface_code.flasq.flasq_model import FLASQSummary  # Import the new summary dataclass
from qualtran.surface_code.flasq.flasq_model import (
    optimistic_FLASQ_costs,  # Import the new instance
)
from qualtran.surface_code.flasq.flasq_model import (
    apply_flasq_cost_model,
    FLASQCostModel,
)
from qualtran.surface_code.flasq.measurement_depth import (  # Import MeasurementDepth
    MeasurementDepth,
    TotalMeasurementDepth,
)
from qualtran.surface_code.flasq.span_counting import (
    BloqWithSpanInfo,
    GateSpan,
    TotalSpanCost,
)
from qualtran.surface_code.flasq.symbols import (
    MIXED_FALLBACK_T_COUNT,
    ROTATION_ERROR,
    T_REACT,
    V_CULT_FACTOR,
)
from qualtran.surface_code.flasq.utils import (  # Needed for the method implementation
    substitute_until_fixed_point,
)
from qualtran.surface_code.flasq.volume_counting import (
    FLASQGateCounts,
    FLASQGateTotals,
)


def test_flasq_cost_model_defaults():
    """Test instantiation with default conservative volumes."""
    model = FLASQCostModel()
    # Test base parameters against their hard-coded conservative values
    assert model.h_volume == 7.0
    assert model.s_volume == 5.5
    assert model.cnot_base_volume == 0.0
    assert model.cz_base_volume == 0.0
    assert model.connect_span_volume == 4.0
    assert model.compute_span_volume == 1.0
    assert model.t_clifford_volume == T_REACT + 6.0
    assert model.t_cultivation_volume == 1.5 * V_CULT_FACTOR
    assert model.toffoli_clifford_volume == 5 * T_REACT + 68.0
    assert model.and_clifford_volume == 2 * T_REACT + 64.0
    assert model.and_dagger_clifford_volume == 0.0

    # Test that derived parameters are calculated correctly from base defaults
    assert model.toffoli_cultivation_volume == 4 * model.t_cultivation_volume
    assert model.and_cultivation_volume == 4 * model.t_cultivation_volume
    assert model.extra_cost_per_t_gate_in_rotation == 2.0
    assert model.extra_cost_per_rotation == 45.0


def test_flasq_cost_model_concrete():
    """Test instantiation with concrete integer volumes."""
    model = FLASQCostModel(
        t_clifford_volume=4,
        t_cultivation_volume=8,  # Provide both parts
        toffoli_clifford_volume=7,
        toffoli_cultivation_volume=14,
        and_clifford_volume=5,
        and_cultivation_volume=10,
        and_dagger_clifford_volume=1.0,
        rz_clifford_volume=1,
        rz_cultivation_volume=2,
        rx_clifford_volume=1.0,
        rx_cultivation_volume=2.0,
        h_volume=0.0,
        s_volume=0.0,
        cnot_base_volume=1.0,
        cz_base_volume=1.0,
        connect_span_volume=0.5,
        compute_span_volume=1.5,

        extra_cost_per_t_gate_in_rotation=1.0,
        extra_cost_per_rotation=10.0,
    )
    assert model.t_clifford_volume == 4
    assert model.t_cultivation_volume == 8
    assert model.toffoli_clifford_volume == 7
    assert model.toffoli_cultivation_volume == 14
    assert model.and_clifford_volume == 5
    assert model.and_cultivation_volume == 10
    assert model.rz_clifford_volume == 1
    assert model.rz_cultivation_volume == 2
    assert model.rx_clifford_volume == 1.0
    assert model.rx_cultivation_volume == 2.0
    assert model.h_volume == 0.0
    assert model.s_volume == 0.0
    assert model.cnot_base_volume == 1.0
    assert model.cz_base_volume == 1.0
    assert model.connect_span_volume == 0.5
    assert model.compute_span_volume == 1.5
    assert model.extra_cost_per_t_gate_in_rotation == 1.0
    assert model.extra_cost_per_rotation == 10.0


def test_calculate_clifford_volume_symbolic():
    """Test Clifford volume calculation with default volumes and symbolic counts."""
    model = FLASQCostModel()  # Default conservative volumes
    N_t = sympy.symbols("N_t")
    N_cnot = sympy.symbols("N_cnot")
    connect_s_total = sympy.symbols("connect_s_total")
    compute_s_total = sympy.symbols("compute_s_total")

    counts = FLASQGateCounts(t=N_t, cnot=N_cnot, hadamard=5)
    span_info = GateSpan(connect_span=connect_s_total, compute_span=compute_s_total)

    # Use the model's default values in the expected volume
    # This test now only checks pure clifford volume.
    expected_volume = (
        N_cnot * model.cnot_base_volume
        + 5 * model.h_volume
        + connect_s_total * model.connect_span_volume
        + compute_s_total * model.compute_span_volume
    )

    total_volume = model.calculate_volume_required_for_clifford_computation(
        counts, span_info
    )
    assert total_volume == expected_volume


def test_calculate_volumes_concrete():
    """Test Clifford and Cultivation volume calculation with concrete volumes and counts."""
    model = FLASQCostModel(
        t_clifford_volume=4,
        t_cultivation_volume=8,  # Set cultivation to 8
        toffoli_clifford_volume=7,
        toffoli_cultivation_volume=14,
        and_clifford_volume=5,
        and_cultivation_volume=10,
        and_dagger_clifford_volume=1.0,
        rz_clifford_volume=1,
        rz_cultivation_volume=2,
        rx_clifford_volume=1.0,
        rx_cultivation_volume=2.0,
        h_volume=0.0,
        s_volume=0.0,  # Explicitly set s_volume
        cnot_base_volume=1.0,  # Use correct attribute name
        cz_base_volume=1.0,  # Use correct attribute name
        connect_span_volume=2.0,
        compute_span_volume=1.0,

        extra_cost_per_t_gate_in_rotation=1.0,
        extra_cost_per_rotation=10.0,
    )
    counts = FLASQGateCounts(
        t=10, cnot=20, hadamard=5, z_rotation=3, cz=2, and_gate=4, and_dagger_gate=1
    )
    span_info = GateSpan(connect_span=15, compute_span=10)

    # Test pure clifford volume
    clifford_volume = model.calculate_volume_required_for_clifford_computation(
        counts, span_info
    )
    # Expected: 20*cnot + 5*h + 2*cz + 1*and_dagger_clifford + 15*connect_span + 10*compute_span
    #         = 20*1 + 5*0 + 2*1 + 1*1 + 15*2 + 10*1 = 20 + 0 + 2 + 1 + 30 + 10 = 63
    assert clifford_volume == pytest.approx(63)

    # Test non-clifford lattice surgery volume
    non_clifford_vol = model.calculate_non_clifford_lattice_surgery_volume(counts)
    # Expected: 10*t + 3*rz + 4*and
    #         = 10*4 + 3*1 + 4*5 = 40 + 3 + 20 = 63.0
    assert non_clifford_vol == pytest.approx(63.0)

    # Test cultivation volume separately
    cultivation_volume = model.calculate_volume_required_for_cultivation(counts)
    # Expected: 10*t_cult + 4*and_cult + 3*rz_cult
    #         = 10*8 + 4*10 + 3*2 = 80 + 40 + 6 = 126
    assert cultivation_volume == pytest.approx(126)


def test_calculate_clifford_volume_mixed():
    """Test Clifford volume calculation with mixed concrete/symbolic volumes and counts."""
    N_t = sympy.symbols("N_t")
    # Use default model, which has symbolic rotation volumes
    model = FLASQCostModel(
        t_clifford_volume=4,  # Concrete Clifford T volume
        connect_span_volume=2,
        compute_span_volume=1,
    )
    counts = FLASQGateCounts(cnot=20)
    span_info = GateSpan(connect_span=15, compute_span=10)

    _conservative_cnot_base_volume = 0.0  # Value from the default constructor

    expected_volume = 20 * _conservative_cnot_base_volume + 15 * 2 + 10 * 1
    total_volume = model.calculate_volume_required_for_clifford_computation(
        counts, span_info
    )
    assert total_volume == pytest.approx(expected_volume)


def test_calculate_volumes_with_unknowns():
    """Test that warnings are issued for volume calculations when unknowns are present."""
    # Use a model with some concrete values, but t_cultivation_volume will be default
    model = FLASQCostModel(
        t_clifford_volume=4, connect_span_volume=2, compute_span_volume=1
    )
    counts_unknown = FLASQGateCounts(
        t=10, bloqs_with_unknown_cost={Ry(sympy.Symbol("theta")): 1}
    )
    span_unknown = GateSpan(
        connect_span=15, compute_span=10, uncounted_bloqs={CNOT(): 1}
    )

    # Test with unknown counts
    with pytest.warns(
        UserWarning, match="pure Clifford volume with unknown FLASQ counts"
    ):
        vol1 = model.calculate_volume_required_for_clifford_computation(
            counts_unknown, GateSpan(connect_span=5, compute_span=5)
        )
    # Test for mathematical equivalence, not exact symbolic form.
    # Should be 0 since T is not a pure clifford.
    assert sympy.simplify(vol1 - (5 * 2 + 5 * 1)) == 0

    # Test with uncounted span
    with pytest.warns(UserWarning, match="Clifford volume with uncounted span bloqs"):
        vol2 = model.calculate_volume_required_for_clifford_computation(
            FLASQGateCounts(t=10), span_unknown
        )
    # Test for mathematical equivalence, not exact symbolic form.
    assert sympy.simplify(vol2 - (15 * 2 + 10 * 1)) == 0

    # Test non-clifford volume warning
    with pytest.warns(UserWarning, match="non-Clifford lattice surgery volume"):
        vol_non_cliff = model.calculate_non_clifford_lattice_surgery_volume(
            counts_unknown
        )
    assert sympy.simplify(vol_non_cliff - 10 * 4) == 0

    # Test cultivation volume warning
    with pytest.warns(
        UserWarning, match="cultivation volume with unknown FLASQ counts"
    ):
        # Value from the default constructor
        _conservative_t_cultivation_volume = 1.5 * V_CULT_FACTOR
        vol_cult = model.calculate_volume_required_for_cultivation(counts_unknown)
    # Test for mathematical equivalence, not exact symbolic form.
    assert sympy.simplify(vol_cult - 10 * _conservative_t_cultivation_volume) == 0

    # Test with both
    # We call the Clifford volume calculation here, which should trigger both warnings
    with pytest.warns(UserWarning) as record:
        vol3 = model.calculate_volume_required_for_clifford_computation(
            counts_unknown, span_unknown
        )
    # Test for mathematical equivalence, not exact symbolic form.
    assert sympy.simplify(vol3 - (15 * 2 + 10 * 1)) == 0
    # Check that two warnings were issued by this specific call
    assert len(record) == 2
    assert "pure Clifford volume with unknown FLASQ counts" in str(record[0].message)
    assert "Clifford volume with uncounted span bloqs" in str(record[1].message)


def test_calculate_volumes_from_circuit_example():
    """Test calculating volumes using counts from a simple circuit."""
    # 1. Build a simple circuit and get counts/span
    bb = BloqBuilder()
    q0 = bb.add_register("q0", 1)
    q1 = bb.add_register("q1", 1)
    q0 = bb.add(Hadamard(), q=q0)
    # Wrap CNOT to give it span info for this test
    q0, q1 = bb.add(
        BloqWithSpanInfo(wrapped_bloq=CNOT(), connect_span=5, compute_span=5),
        ctrl=q0,
        target=q1,
    )
    q1 = bb.add(Hadamard(), q=q1)
    cbloq = bb.finalize(q0=q0, q1=q1)

    flasq_counts = get_cost_value(cbloq, FLASQGateTotals())
    span_info = get_cost_value(cbloq, TotalSpanCost())

    assert flasq_counts == FLASQGateCounts(hadamard=2, cnot=1)
    assert span_info == GateSpan(connect_span=5, compute_span=5)

    # 2. Define a concrete volume model
    model = FLASQCostModel(
        t_clifford_volume=4,  # Example values
        t_cultivation_volume=8,
        toffoli_clifford_volume=7,
        toffoli_cultivation_volume=14,
        and_clifford_volume=5,
        and_cultivation_volume=10,
        rz_clifford_volume=1,
        rz_cultivation_volume=2,
        rx_clifford_volume=1.0,
        rx_cultivation_volume=2.0,
        h_volume=0.0,
        s_volume=0.0,  # Explicitly set s_volume
        cnot_base_volume=1.0,  # Use correct attribute name
        cz_base_volume=1.0,  # Use correct attribute name
        connect_span_volume=2.0,
        compute_span_volume=1.0,

        extra_cost_per_t_gate_in_rotation=1.0,
        extra_cost_per_rotation=10.0,
    )

    # 3. Calculate total Clifford volume
    total_clifford_volume = model.calculate_volume_required_for_clifford_computation(
        flasq_counts, span_info
    )

    # Expected Clifford Volume: (2*H_VOL) + (1*CNOT_BASE_VOL) + (5*CONNECT_VOL) + (5*COMPUTE_VOL)
    #         = (2*0) + (1*1) + (5*2) + (5*1) = 0 + 1 + 10 + 5 = 16
    assert total_clifford_volume == 16.0

    # Cultivation volume should be 0 for H and CNOT
    assert model.calculate_volume_required_for_cultivation(flasq_counts) == 0.0


# --- Tests for FLASQSummary.resolve_symbols ---


def test_flasq_gate_counts_total_rotations():
    """Test the total_rotations property of FLASQGateCounts."""
    # Test with concrete numbers
    counts1 = FLASQGateCounts(x_rotation=3, z_rotation=5)
    assert counts1.total_rotations == 8

    # Test with zero rotations
    counts2 = FLASQGateCounts(t=10)
    assert counts2.total_rotations == 0

    # Test with symbolic numbers
    N_x = sympy.Symbol("N_x")
    N_z = sympy.Symbol("N_z")
    counts3 = FLASQGateCounts(x_rotation=N_x, z_rotation=N_z)
    assert counts3.total_rotations == N_x + N_z

    # Test with one symbolic, one concrete
    counts4 = FLASQGateCounts(x_rotation=N_x, z_rotation=2)
    assert counts4.total_rotations == N_x + 2


def test_resolve_symbols_basic():
    """Test basic substitution and type conversion in resolve_symbols."""
    N = sympy.Symbol("N")
    M = sympy.Symbol("M")
    P = sympy.Symbol("P")
    # V_CULT_FACTOR is already imported
    err_sym = sympy.Symbol("err_sym")  # Renamed to avoid conflict

    symbolic_summary = FLASQSummary(
        total_clifford_volume=N * 10 + M,
        total_depth=N / 5.0,
        n_algorithmic_qubits=N,
        n_fluid_ancilla=M + 1,
        total_t_count=P / err_sym,
        total_rotation_count=M,
        # New fields
        measurement_depth_val=N / 10.0,
        scaled_measurement_depth=N / 10.0,
        volume_limited_depth=M * 2 + N / 20.0,  # Mix N and M
        total_computational_volume=N * 8 + M * 3,
        idling_volume=N * (N / 5.0),  # N * total_depth
        clifford_computational_volume=N * 2 + M,
        non_clifford_lattice_surgery_volume=P,
        cultivation_volume=P + M + V_CULT_FACTOR * 1.5,  # Added V_CULT_FACTOR term
        # This is wrong, but the test is about resolution, not correctness of this value.
        total_spacetime_volume=N * 2
        + M
        + P
        + P
        + M
        + V_CULT_FACTOR * 1.5
        + N * (N / 5.0),
    )

    assumptions = {
        N: 100,
        P: 50,
        err_sym: 0.01,
        V_CULT_FACTOR: 6,
    }  # M is left unresolved

    resolved_summary = symbolic_summary.resolve_symbols(frozendict(assumptions))

    # Check it's a new instance
    assert resolved_summary is not symbolic_summary

    # Check resolved values and types
    assert resolved_summary.n_algorithmic_qubits == 100
    assert isinstance(resolved_summary.n_algorithmic_qubits, int)

    assert resolved_summary.total_depth == 20.0
    assert isinstance(resolved_summary.total_depth, float)

    assert resolved_summary.total_t_count == 5000.0
    assert isinstance(resolved_summary.total_t_count, float)

    # Check unresolved symbols remain
    assert resolved_summary.total_clifford_volume == 1000 + M
    assert isinstance(resolved_summary.total_clifford_volume, sympy.Expr)

    assert resolved_summary.n_fluid_ancilla == M + 1
    assert isinstance(resolved_summary.n_fluid_ancilla, sympy.Expr)

    assert resolved_summary.total_rotation_count == M
    assert isinstance(resolved_summary.total_rotation_count, sympy.Expr)

    # Check new fields
    assert resolved_summary.measurement_depth_val == 10.0
    assert isinstance(resolved_summary.measurement_depth_val, float)

    assert resolved_summary.volume_limited_depth == M * 2 + 5.0
    assert isinstance(resolved_summary.volume_limited_depth, sympy.Expr)

    assert resolved_summary.total_computational_volume == 800 + M * 3
    assert isinstance(resolved_summary.total_computational_volume, sympy.Expr)

    assert resolved_summary.idling_volume == 2000.0
    assert isinstance(resolved_summary.idling_volume, float)

    assert resolved_summary.clifford_computational_volume == 200 + M
    assert isinstance(resolved_summary.clifford_computational_volume, sympy.Expr)

    assert resolved_summary.non_clifford_lattice_surgery_volume == 50
    assert isinstance(resolved_summary.non_clifford_lattice_surgery_volume, int)

    assert resolved_summary.cultivation_volume == 50 + M + (6 * 1.5)  # 50 + M + 9
    assert isinstance(resolved_summary.cultivation_volume, sympy.Expr)

    assert resolved_summary.total_spacetime_volume == 200 + M + 50 + 50 + M + 9 + 2000.0
    assert isinstance(resolved_summary.total_spacetime_volume, sympy.Expr)
    # 2309 + 2*M
    assert resolved_summary.total_spacetime_volume == 2309.0 + 2 * M


def test_resolve_symbols_nested():  # sourcery skip: extract-method
    """Test resolution of nested symbolic dependencies."""
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    z = sympy.Symbol("z")

    # Use only one field for simplicity
    summary = FLASQSummary(
        total_depth=x + 1.0,
        total_clifford_volume=0.0,
        n_algorithmic_qubits=0,
        n_fluid_ancilla=0,
        total_t_count=0.0,
        total_rotation_count=0.0,
        # New fields (can be zero/float as they are not the focus of this test)
        measurement_depth_val=0,
        volume_limited_depth=0,
        scaled_measurement_depth=0,
        total_computational_volume=0,
        idling_volume=0,
        clifford_computational_volume=0,
        non_clifford_lattice_surgery_volume=0,
        cultivation_volume=0,
        total_spacetime_volume=0,
    )

    nested_assumptions = {x: y * 2.0, y: z + 1.0, z: 5.0}

    resolved_summary = summary.resolve_symbols(frozendict(nested_assumptions))

    # x -> y*2 -> (z+1)*2 -> (5+1)*2 = 12. total_depth = x+1 = 13.
    assert resolved_summary.total_depth == 13.0
    assert isinstance(resolved_summary.total_depth, float)
    assert resolved_summary.total_spacetime_volume == 0.0


def test_resolve_symbols_no_symbols():
    """Test calling resolve_symbols on a summary with concrete values."""
    concrete_summary = FLASQSummary(
        total_clifford_volume=1000.0,
        total_depth=20.0,
        n_algorithmic_qubits=100,
        n_fluid_ancilla=50,
        total_t_count=5000.0,
        total_rotation_count=0.0,
        # New fields with concrete values
        measurement_depth_val=10.0,
        volume_limited_depth=5.0,
        scaled_measurement_depth=10.0,
        total_computational_volume=250.0,
        idling_volume=2000.0,  # 100 * 20
        clifford_computational_volume=-1000.0,  # 1000 - 2000
        non_clifford_lattice_surgery_volume=0.0,
        cultivation_volume=170.0,
        total_spacetime_volume=1170.0,  # -1000 + 170 + 2000
    )

    assumptions = {sympy.Symbol("N"): 10}  # Symbol not present in summary

    resolved_concrete = concrete_summary.resolve_symbols(frozendict(assumptions))

    # Should return a new object with the same concrete values
    assert resolved_concrete == concrete_summary
    assert resolved_concrete is not concrete_summary  # Ensure it's a new instance

def test_regular_spacetime_volume_property():
    """Test the regular_spacetime_volume property of FLASQSummary."""
    summary = FLASQSummary(
        total_clifford_volume=1000.0,
        total_depth=20.0,
        n_algorithmic_qubits=100,
        n_fluid_ancilla=50,
        total_t_count=5000.0,
        total_rotation_count=0.0,
        measurement_depth_val=10.0,
        volume_limited_depth=5.0,
        scaled_measurement_depth=10.0,
        total_computational_volume=250.0,
        idling_volume=2000.0,
        clifford_computational_volume=-1000.0,
        non_clifford_lattice_surgery_volume=0.0,
        cultivation_volume=170.0,
        total_spacetime_volume=1170.0,
    )
    assert summary.regular_spacetime_volume == 1000.0  # 1170 - 170


# --- Basic Test for apply_flasq_cost_model ---


def test_apply_flasq_cost_model_basic():
    """Test apply_flasq_cost_model and verify all FLASQSummary fields."""
    # 1. Define a concrete FLASQCostModel
    model = FLASQCostModel(  # All values as floats
        t_clifford_volume=2.0,
        t_cultivation_volume=8.0,  # type: ignore[misc]
        toffoli_clifford_volume=10.0,
        toffoli_cultivation_volume=32.0,  # type: ignore[misc]
        and_clifford_volume=5.0,
        and_cultivation_volume=10.0,  # type: ignore[misc]
        rz_clifford_volume=5.0,
        rz_cultivation_volume=15.0,  # type: ignore[misc]
        rx_clifford_volume=5.0,
        rx_cultivation_volume=15.0,
        h_volume=1.0,
        s_volume=1.0,
        cnot_base_volume=2.0,
        cz_base_volume=2.0,
        connect_span_volume=1.0,
        compute_span_volume=1.0,

        extra_cost_per_t_gate_in_rotation=1.0,
        extra_cost_per_rotation=10.0,
    )

    # 2. Define inputs for apply_flasq_cost_model
    n_algorithmic_qubits_val = 10
    n_total_logical_qubits_val = 110  # Must be > n_algorithmic_qubits
    counts_val = FLASQGateCounts(
        t=5,
        toffoli=2,
        z_rotation=3,
        x_rotation=1,
        hadamard=4,
        s_gate=2,
        cnot=6,
        cz=1,
        and_gate=1,
        and_dagger_gate=1,
    )
    span_info_val = GateSpan(connect_span=20, compute_span=10)
    measurement_depth_val_obj = MeasurementDepth(depth=50)

    # 3. Call apply_flasq_cost_model
    summary = apply_flasq_cost_model(
        model=model,
        n_total_logical_qubits=n_total_logical_qubits_val,
        qubit_counts=n_algorithmic_qubits_val,
        counts=counts_val,
        span_info=span_info_val,
        measurement_depth=measurement_depth_val_obj,
        logical_timesteps_per_measurement=1,
    )

    # 4. Calculate expected values for all fields based on apply_flasq_cost_model logic
    expected_pure_clifford_volume = (
        counts_val.hadamard * model.h_volume
        + counts_val.s_gate * model.s_volume
        + counts_val.cnot * model.cnot_base_volume
        + counts_val.cz * model.cz_base_volume
        + counts_val.and_dagger_gate * model.and_dagger_clifford_volume
        + span_info_val.connect_span * model.connect_span_volume
        + span_info_val.compute_span * model.compute_span_volume
    )  # 4*1+2*1+6*2+1*2+1*0+20*1+10*1 = 4+2+12+2+0+20+10 = 50.0

    expected_non_clifford_lattice_surgery_vol = (
        counts_val.t * model.t_clifford_volume
        + counts_val.toffoli * model.toffoli_clifford_volume
        + counts_val.z_rotation * model.rz_clifford_volume
        + counts_val.x_rotation * model.rx_clifford_volume
        + counts_val.and_gate * model.and_clifford_volume
    )  # 5*2+2*10+3*5+1*5+1*5 = 10+20+15+5+5 = 55

    expected_cultivation_volume = (
        counts_val.t * model.t_cultivation_volume
        + counts_val.toffoli * model.toffoli_cultivation_volume
        + counts_val.and_gate * model.and_cultivation_volume
        # and_dagger_gate has no cultivation volume
        + counts_val.z_rotation * model.rz_cultivation_volume
        + counts_val.x_rotation * model.rx_cultivation_volume
    )  # 5*8 + 2*32 + 1*10 + 3*15 + 1*15 = 40+64+10+45+15 = 174

    expected_total_computational_volume = (
        expected_pure_clifford_volume
        + expected_non_clifford_lattice_surgery_vol
        + expected_cultivation_volume
    )
    n_fluid_ancilla_val = n_total_logical_qubits_val - n_algorithmic_qubits_val
    expected_volume_limited_depth = (
        expected_total_computational_volume / n_fluid_ancilla_val
    )
    expected_total_depth = sympy.Max(
        measurement_depth_val_obj.depth * 1.0, expected_volume_limited_depth
    )
    expected_idling_volume = n_algorithmic_qubits_val * expected_total_depth
    expected_total_clifford_volume = (
        expected_pure_clifford_volume + expected_idling_volume
    )

    expected_total_t_val = (
        counts_val.t
        + counts_val.toffoli * 4
        + counts_val.and_gate * 4
        + MIXED_FALLBACK_T_COUNT * (counts_val.z_rotation + counts_val.x_rotation)
    )
    expected_total_rotation_val = counts_val.x_rotation + counts_val.z_rotation
    expected_total_spacetime_volume = (
        expected_total_computational_volume + expected_idling_volume
    )

    # 5. Assert all fields in the summary
    assert summary.clifford_computational_volume == pytest.approx(
        expected_pure_clifford_volume
    )
    assert summary.non_clifford_lattice_surgery_volume == pytest.approx(
        expected_non_clifford_lattice_surgery_vol
    )
    assert summary.cultivation_volume == pytest.approx(expected_cultivation_volume)
    assert summary.measurement_depth_val == pytest.approx(
        measurement_depth_val_obj.depth
    )
    assert summary.total_computational_volume == pytest.approx(
        expected_total_computational_volume
    )
    assert summary.volume_limited_depth == pytest.approx(expected_volume_limited_depth)
    assert summary.idling_volume == pytest.approx(expected_idling_volume)
    assert summary.total_clifford_volume == pytest.approx(
        expected_total_clifford_volume
    )
    assert summary.total_depth == pytest.approx(expected_total_depth)
    assert summary.n_algorithmic_qubits == n_algorithmic_qubits_val
    assert summary.n_fluid_ancilla == n_fluid_ancilla_val
    assert summary.total_t_count == pytest.approx(expected_total_t_val)
    assert summary.total_rotation_count == expected_total_rotation_val
    assert summary.total_spacetime_volume == pytest.approx(
        expected_total_spacetime_volume
    )


def test_apply_flasq_cost_model_invariants():
    """Test invariants between fields in the FLASQSummary from apply_flasq_cost_model."""
    # 1. Define a concrete FLASQCostModel with non-zero values
    model = FLASQCostModel(  # All values as floats
        t_clifford_volume=2.0,
        t_cultivation_volume=8.0,  # type: ignore[misc]
        toffoli_clifford_volume=10.0,
        toffoli_cultivation_volume=32.0,  # type: ignore[misc]
        and_clifford_volume=5.0,
        and_cultivation_volume=10.0,  # type: ignore[misc]
        rz_clifford_volume=5.0,
        rz_cultivation_volume=15.0,  # type: ignore[misc]
        rx_clifford_volume=5.0,
        rx_cultivation_volume=15.0,
        h_volume=1.0,
        s_volume=1.0,
        cnot_base_volume=2.0,
        cz_base_volume=2.0,
        connect_span_volume=1.0,
        compute_span_volume=1.0,

        extra_cost_per_t_gate_in_rotation=1.0,
        extra_cost_per_rotation=10.0,
    )

    # 2. Define concrete inputs
    n_algorithmic_qubits_val = 10
    n_total_logical_qubits_val = 110  # Ensure non-zero fluid ancillas
    counts_val = FLASQGateCounts(
        t=5, toffoli=2, z_rotation=3, x_rotation=1, hadamard=4, cnot=6
    )
    span_info_val = GateSpan(connect_span=20, compute_span=10)
    measurement_depth_obj = MeasurementDepth(depth=50)

    # 3. Call apply_flasq_cost_model
    summary = apply_flasq_cost_model(
        model=model,
        n_total_logical_qubits=n_total_logical_qubits_val,
        qubit_counts=n_algorithmic_qubits_val,
        counts=counts_val,
        span_info=span_info_val,
        measurement_depth=measurement_depth_obj,
        logical_timesteps_per_measurement=1,
    )

    # 4. Verify Invariants
    # Invariant 1: Total Computational Volume
    assert summary.total_computational_volume == (
        summary.clifford_computational_volume
        + summary.non_clifford_lattice_surgery_volume
        + summary.cultivation_volume
    )

    # Invariant 2: Volume-Limited Depth
    # Ensure n_fluid_ancilla is not zero before division, which it is by design here.
    assert summary.volume_limited_depth == (
        summary.total_computational_volume / summary.n_fluid_ancilla
    )

    # Invariant 3: Total Depth
    assert summary.total_depth == sympy.Max(  # type: ignore[comparison-fn]
        summary.measurement_depth_val, summary.volume_limited_depth
    )

    # Invariant 4: Idling Volume
    assert summary.idling_volume == summary.n_algorithmic_qubits * summary.total_depth

    # Invariant 5: Total Clifford Volume
    assert summary.total_clifford_volume == (
        summary.clifford_computational_volume + summary.idling_volume
    )

    # Invariant 6: Total Spacetime Volume
    assert summary.total_spacetime_volume == (
        summary.total_computational_volume + summary.idling_volume
    )


def test_resolve_symbols_empty_assumptions():
    """Test calling resolve_symbols with an empty assumptions dictionary."""
    N = sympy.Symbol("N")
    symbolic_summary = FLASQSummary(
        total_clifford_volume=N * 10,
        total_depth=N / 5,
        n_algorithmic_qubits=N,
        n_fluid_ancilla=N + 1,
        total_t_count=N * 2.0,
        total_rotation_count=0.0,
        # New fields with symbolic values
        measurement_depth_val=N / 10.0,
        scaled_measurement_depth=N / 10.0,
        volume_limited_depth=(N * 8 + N * 4)
        / (N + 1.0),  # (cliff_comp + cult) / fluid_ancilla
        total_computational_volume=N * 8 + N * 4,  # cliff_comp + cult
        idling_volume=N * (N / 5),  # data_qubits * total_depth
        clifford_computational_volume=N * 8,
        non_clifford_lattice_surgery_volume=0,
        cultivation_volume=N * 4,
        total_spacetime_volume=N * 8 + N * 4 + N * (N / 5),
    )
    resolved_empty = symbolic_summary.resolve_symbols(frozendict({}))
    # Values should remain symbolic. Direct comparison might fail if sympy
    # simplifies expressions (e.g. N*8 + N*4 -> 12*N). Instead, we check
    # that the difference between each field simplifies to zero.
    for field in attrs.fields(FLASQSummary):
        original_val = getattr(symbolic_summary, field.name)
        resolved_val = getattr(resolved_empty, field.name)
        assert sympy.Float(sympy.simplify(original_val - resolved_val)) == 0.0


def test_exported_flasq_cost_models():
    """Test the pre-instantiated conservative and optimistic FLASQCostModel instances."""
    # Conservative assertions (values from flasq_model.py)
    assert conservative_FLASQ_costs.h_volume == 7.0
    assert conservative_FLASQ_costs.s_volume == 5.5
    assert conservative_FLASQ_costs.t_cultivation_volume == 1.5 * V_CULT_FACTOR
    assert conservative_FLASQ_costs.t_clifford_volume == T_REACT + 6.0
    assert conservative_FLASQ_costs.cnot_base_volume == 0.0
    assert conservative_FLASQ_costs.connect_span_volume == 4.0
    assert conservative_FLASQ_costs.compute_span_volume == 1.0

    assert conservative_FLASQ_costs.toffoli_cultivation_volume == 6.0 * V_CULT_FACTOR
    assert conservative_FLASQ_costs.toffoli_clifford_volume == 5 * T_REACT + 68.0
    assert conservative_FLASQ_costs.and_cultivation_volume == 4 * (1.5 * V_CULT_FACTOR)
    assert conservative_FLASQ_costs.and_clifford_volume == 2 * T_REACT + 64.0
    assert conservative_FLASQ_costs.and_dagger_clifford_volume == 0.0
    assert conservative_FLASQ_costs.extra_cost_per_t_gate_in_rotation == 2.0
    assert conservative_FLASQ_costs.extra_cost_per_rotation == 45.0

    # Optimistic assertions (values from flasq_model.py)
    # Base parameters
    assert optimistic_FLASQ_costs.h_volume == 1.5
    assert optimistic_FLASQ_costs.s_volume == 1.5
    assert optimistic_FLASQ_costs.t_cultivation_volume == V_CULT_FACTOR
    assert optimistic_FLASQ_costs.t_clifford_volume == T_REACT + 2.5
    assert optimistic_FLASQ_costs.cnot_base_volume == 0.0
    assert optimistic_FLASQ_costs.connect_span_volume == 1.0
    assert optimistic_FLASQ_costs.compute_span_volume == 1.0

    assert optimistic_FLASQ_costs.toffoli_clifford_volume == 5 * T_REACT + 39.0
    assert optimistic_FLASQ_costs.and_clifford_volume == 2 * T_REACT + 36.0
    assert optimistic_FLASQ_costs.and_dagger_clifford_volume == 0.0
    assert optimistic_FLASQ_costs.extra_cost_per_t_gate_in_rotation == 1.0
    assert optimistic_FLASQ_costs.extra_cost_per_rotation == 12.0

    # Derived parameters
    assert (
        optimistic_FLASQ_costs.toffoli_cultivation_volume
        == 4 * optimistic_FLASQ_costs.t_cultivation_volume
    )
    assert (
        optimistic_FLASQ_costs.and_cultivation_volume
        == 4 * optimistic_FLASQ_costs.t_cultivation_volume
    )


def test_apply_flasq_cost_model_with_defaults_and_resolution():
    """Test apply_flasq_cost_model with conservative_FLASQ_costs and symbol resolution."""
    n_data_qubits_val = 10
    n_fluid_ancilla_val = 100
    counts_val = FLASQGateCounts(
        t=5, toffoli=2, z_rotation=3, cnot=6, hadamard=4, s_gate=1, and_gate=1
    )
    span_info_val = GateSpan(connect_span=20, compute_span=10)
    measurement_depth_obj = MeasurementDepth(depth=50.0)

    summary_symbolic = apply_flasq_cost_model(
        model=conservative_FLASQ_costs,
        n_total_logical_qubits=n_data_qubits_val + n_fluid_ancilla_val,
        qubit_counts=n_data_qubits_val,
        counts=counts_val,
        span_info=span_info_val,
        measurement_depth=measurement_depth_obj,
        logical_timesteps_per_measurement=1,
    )

    # Check some symbolic fields before resolution
    assert V_CULT_FACTOR in summary_symbolic.cultivation_volume.free_symbols
    assert V_CULT_FACTOR in summary_symbolic.total_spacetime_volume.free_symbols
    assert ROTATION_ERROR in summary_symbolic.total_spacetime_volume.free_symbols
    # T_REACT is now in the clifford part of rotations
    assert T_REACT in summary_symbolic.non_clifford_lattice_surgery_volume.free_symbols
    assert T_REACT in summary_symbolic.total_spacetime_volume.free_symbols

    resolved_summary = summary_symbolic.resolve_symbols(
        frozendict(
            {
                V_CULT_FACTOR: 6.0,
                ROTATION_ERROR: 1e-3,
                T_REACT: 1.0,
            }
        )
    )

    # Instead of recalculating all by hand, check a few key ones are now numbers
    assert isinstance(resolved_summary.cultivation_volume, float)
    assert isinstance(resolved_summary.clifford_computational_volume, float)
    assert isinstance(resolved_summary.total_depth, float)
    assert isinstance(resolved_summary.total_spacetime_volume, float)
    assert resolved_summary.cultivation_volume > 0
    assert resolved_summary.clifford_computational_volume > 0
    assert resolved_summary.total_spacetime_volume > 0


def test_flasq_summary_is_limited_properties_on_symbolic():
    """Test that is_volume_limited and is_reaction_limited raise errors on symbolic summaries."""
    v_depth = sympy.Symbol("v_depth")
    m_depth = sympy.Symbol("m_depth")

    symbolic_summary = FLASQSummary(
        total_clifford_volume=0,
        total_depth=0,
        n_algorithmic_qubits=0,
        n_fluid_ancilla=0,
        total_t_count=0,
        total_rotation_count=0,
        measurement_depth_val=0,
        volume_limited_depth=v_depth,
        scaled_measurement_depth=m_depth,
        total_computational_volume=0,
        idling_volume=0,
        clifford_computational_volume=0,
        non_clifford_lattice_surgery_volume=0,
        cultivation_volume=0,
        total_spacetime_volume=0,
    )

    with pytest.raises(
        ValueError, match="Cannot determine if summary is volume-limited"
    ):
        _ = symbolic_summary.is_volume_limited

    with pytest.raises(
        ValueError, match="Cannot determine if summary is reaction-limited"
    ):
        _ = symbolic_summary.is_reaction_limited

    # Test with one symbolic, one concrete
    concrete_summary = attrs.evolve(symbolic_summary, scaled_measurement_depth=100.0)

    with pytest.raises(
        ValueError, match="Cannot determine if summary is volume-limited"
    ):
        _ = concrete_summary.is_volume_limited

    with pytest.raises(
        ValueError, match="Cannot determine if summary is reaction-limited"
    ):
        _ = concrete_summary.is_reaction_limited

    # Test with concrete values - should not raise
    resolved_summary = concrete_summary.resolve_symbols(frozendict({v_depth: 50.0}))
    assert not resolved_summary.is_volume_limited
    assert resolved_summary.is_reaction_limited


# --- End-to-End Example Tests ---


def test_end_to_end_summary_from_custom_circuit():
    """An end-to-end example of getting a FLASQSummary from a Cirq circuit.

    This test serves as a "how-to" guide, demonstrating the full pipeline from
    a `cirq.Circuit` to a final `FLASQSummary` with resolved symbolic costs.
    """
    # 1. Define a cirq.Circuit. We use GridQubits as they are required for
    #    span calculations.
    q = cirq.GridQubit.rect(1, 4)
    circuit = cirq.Circuit(
        cirq.H(q[0]),
        cirq.S(q[1]),
        cirq.CNOT(q[0], q[1]),
        cirq.TOFFOLI(q[0], q[1], q[2]),
        cirq.rz(0.123).on(q[3]),
    )

    # 2. Define parameters for the cost model application.
    n_algorithmic_qubits = len(circuit.all_qubits())
    n_total_logical_qubits = n_algorithmic_qubits + 20
    logical_timesteps_per_measurement = 1.0
    # Assumptions to resolve symbolic costs from the conservative model.
    assumptions = frozendict({ROTATION_ERROR: 1e-3, V_CULT_FACTOR: 6.0, T_REACT: 1.0})

    # 3. Convert the cirq.Circuit into a qualtran CompositeBloq. This is the
    #    main entry point to the FLASQ analysis framework.
    cbloq, _ = convert_circuit_for_flasq_analysis(circuit)

    # 4. Count the abstract resources from the CompositeBloq.
    flasq_counts = get_cost_value(cbloq, FLASQGateTotals())
    span_info = get_cost_value(cbloq, TotalSpanCost())
    qubit_counts = get_cost_value(cbloq, QubitCount())
    assert qubit_counts == n_algorithmic_qubits

    # We must provide a rotation_error to get a concrete measurement depth.
    rotation_depth_val = substitute_until_fixed_point(
        MIXED_FALLBACK_T_COUNT, assumptions, try_make_number=True
    )
    measurement_depth = get_cost_value(
        cbloq, TotalMeasurementDepth(rotation_depth=rotation_depth_val)
    )

    # 5. Apply the cost model to get a FLASQSummary. Here, we use the default
    #    `conservative_FLASQ_costs` model, which has symbolic parameters.
    summary_symbolic = apply_flasq_cost_model(
        model=conservative_FLASQ_costs,
        n_total_logical_qubits=n_total_logical_qubits,
        qubit_counts=qubit_counts,
        counts=flasq_counts,
        span_info=span_info,
        measurement_depth=measurement_depth,
        logical_timesteps_per_measurement=logical_timesteps_per_measurement,
    )

    # 6. Resolve the symbolic summary into a concrete one using our assumptions.
    summary_resolved = summary_symbolic.resolve_symbols(assumptions)

    # This is a regression test against a golden value.
    # See the detailed calculation in the PR description or commit message that
    # updated this value. The T-count formula for rotations was updated, leading
    # to this new value.
    assert np.isclose(summary_resolved.total_spacetime_volume, 436.86, atol=0.1)


@pytest.mark.slow
def test_end_to_end_summary_from_hwp_circuit_repr():
    """An end-to-end example using a complex circuit stored as a `repr` string.

    This test demonstrates analysis on a more realistic circuit and shows how
    to use a custom FLASQCostModel and ignore measurement depth.
    """
    # 1. Store a complex circuit as its `repr()` string. This is useful for
    #    vendoring large, pre-generated circuits into tests without cluttering them.
    #    This specific circuit is a decomposition of HammingWeightPhasing on 7 qubits.
    hwp_circuit_repr = "cirq.Circuit([cirq.Moment([cirq.CNOT(cirq.GridQubit(0, 6), cirq.GridQubit(0, 5)), cirq.H(cirq.GridQubit(-1, 0)), cirq.H(cirq.GridQubit(-1, 1)), cirq.H(cirq.GridQubit(-1, 2)), cirq.H(cirq.GridQubit(-1, 3))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(0, 6), cirq.GridQubit(0, 4)), cirq.T(cirq.GridQubit(-1, 0)), cirq.T(cirq.GridQubit(-1, 1)), cirq.T(cirq.GridQubit(-1, 2)), cirq.T(cirq.GridQubit(-1, 3))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(0, 5), cirq.GridQubit(-1, 0))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(0, 4), cirq.GridQubit(-1, 0))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(-1, 0), cirq.GridQubit(0, 5))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(-1, 0), cirq.GridQubit(0, 4)), (cirq.T**-1).on(cirq.GridQubit(0, 5))]), cirq.Moment([(cirq.T**-1).on(cirq.GridQubit(0, 4)), cirq.T(cirq.GridQubit(-1, 0))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(-1, 0), cirq.GridQubit(0, 5))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(-1, 0), cirq.GridQubit(0, 4)), cirq.CNOT(cirq.GridQubit(0, 6), cirq.GridQubit(0, 5))]), cirq.Moment([cirq.H(cirq.GridQubit(-1, 0)), cirq.CNOT(cirq.GridQubit(0, 5), cirq.GridQubit(0, 4))]), cirq.Moment([cirq.S(cirq.GridQubit(-1, 0)), cirq.CNOT(cirq.GridQubit(0, 4), cirq.GridQubit(0, 3))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(0, 6), cirq.GridQubit(-1, 0)), cirq.CNOT(cirq.GridQubit(0, 4), cirq.GridQubit(0, 2)), cirq.CNOT(cirq.GridQubit(0, 3), cirq.GridQubit(-1, 1))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(0, 2), cirq.GridQubit(-1, 1))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(-1, 1), cirq.GridQubit(0, 3))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(-1, 1), cirq.GridQubit(0, 2)), (cirq.T**-1).on(cirq.GridQubit(0, 3))]), cirq.Moment([(cirq.T**-1).on(cirq.GridQubit(0, 2)), cirq.T(cirq.GridQubit(-1, 1))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(-1, 1), cirq.GridQubit(0, 3))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(-1, 1), cirq.GridQubit(0, 2)), cirq.CNOT(cirq.GridQubit(0, 4), cirq.GridQubit(0, 3))]), cirq.Moment([cirq.H(cirq.GridQubit(-1, 1)), cirq.CNOT(cirq.GridQubit(0, 3), cirq.GridQubit(0, 2))]), cirq.Moment([cirq.S(cirq.GridQubit(-1, 1)), cirq.CNOT(cirq.GridQubit(0, 2), cirq.GridQubit(0, 1))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(0, 4), cirq.GridQubit(-1, 1)), cirq.CNOT(cirq.GridQubit(0, 2), cirq.GridQubit(0, 0)), cirq.CNOT(cirq.GridQubit(0, 1), cirq.GridQubit(-1, 2))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(0, 0), cirq.GridQubit(-1, 2)), cirq.CNOT(cirq.GridQubit(-1, 0), cirq.GridQubit(-1, 1))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(-1, 2), cirq.GridQubit(0, 1)), cirq.CNOT(cirq.GridQubit(-1, 1), cirq.GridQubit(-1, 3))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(-1, 2), cirq.GridQubit(0, 0)), (cirq.T**-1).on(cirq.GridQubit(0, 1))]), cirq.Moment([(cirq.T**-1).on(cirq.GridQubit(0, 0)), cirq.T(cirq.GridQubit(-1, 2))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(-1, 2), cirq.GridQubit(0, 1))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(-1, 2), cirq.GridQubit(0, 0)), cirq.CNOT(cirq.GridQubit(0, 2), cirq.GridQubit(0, 1))]), cirq.Moment([cirq.H(cirq.GridQubit(-1, 2)), cirq.CNOT(cirq.GridQubit(0, 1), cirq.GridQubit(0, 0))]), cirq.Moment([cirq.S(cirq.GridQubit(-1, 2)), cirq.CNOT(cirq.GridQubit(0, 0), cirq.GridQubit(-1, 6))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(0, 2), cirq.GridQubit(-1, 2)), (cirq.Z**0.03915211600060625).on(cirq.GridQubit(-1, 6))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(-1, 0), cirq.GridQubit(-1, 2)), cirq.CNOT(cirq.GridQubit(0, 0), cirq.GridQubit(-1, 6))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(-1, 2), cirq.GridQubit(-1, 3)), cirq.CNOT(cirq.GridQubit(0, 1), cirq.GridQubit(0, 0))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(-1, 3), cirq.GridQubit(-1, 1))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(-1, 3), cirq.GridQubit(-1, 2)), (cirq.T**-1).on(cirq.GridQubit(-1, 1))]), cirq.Moment([(cirq.T**-1).on(cirq.GridQubit(-1, 2)), cirq.T(cirq.GridQubit(-1, 3))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(-1, 3), cirq.GridQubit(-1, 1))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(-1, 3), cirq.GridQubit(-1, 2)), cirq.CNOT(cirq.GridQubit(-1, 0), cirq.GridQubit(-1, 1))]), cirq.Moment([cirq.H(cirq.GridQubit(-1, 3)), cirq.CNOT(cirq.GridQubit(-1, 1), cirq.GridQubit(-1, 2))]), cirq.Moment([cirq.S(cirq.GridQubit(-1, 3)), cirq.CNOT(cirq.GridQubit(-1, 2), cirq.GridQubit(-1, 5))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(-1, 0), cirq.GridQubit(-1, 3)), (cirq.Z**0.0783042320012125).on(cirq.GridQubit(-1, 5))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(-1, 3), cirq.GridQubit(-1, 4)), cirq.CNOT(cirq.GridQubit(-1, 2), cirq.GridQubit(-1, 5))]), cirq.Moment([(cirq.Z**0.156608464002425).on(cirq.GridQubit(-1, 4)), cirq.CNOT(cirq.GridQubit(-1, 1), cirq.GridQubit(-1, 2))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(-1, 3), cirq.GridQubit(-1, 4))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(-1, 0), cirq.GridQubit(-1, 3))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(-1, 0), cirq.GridQubit(-1, 1))]), cirq.Moment([And(cv1=1, cv2=1, uncompute=True).on(cirq.GridQubit(-1, 1), cirq.GridQubit(-1, 2), cirq.GridQubit(-1, 3))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(-1, 0), cirq.GridQubit(-1, 2))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(0, 2), cirq.GridQubit(-1, 2)), cirq.CNOT(cirq.GridQubit(-1, 0), cirq.GridQubit(-1, 1))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(0, 2), cirq.GridQubit(0, 1)), cirq.CNOT(cirq.GridQubit(0, 4), cirq.GridQubit(-1, 1)), cirq.CNOT(cirq.GridQubit(0, 6), cirq.GridQubit(-1, 0))]), cirq.Moment([And(cv1=1, cv2=1, uncompute=True).on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 0), cirq.GridQubit(-1, 2))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(0, 2), cirq.GridQubit(0, 0))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(0, 2), cirq.GridQubit(0, 1))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(0, 3), cirq.GridQubit(0, 2))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(0, 4), cirq.GridQubit(0, 3))]), cirq.Moment([And(cv1=1, cv2=1, uncompute=True).on(cirq.GridQubit(0, 3), cirq.GridQubit(0, 2), cirq.GridQubit(-1, 1))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(0, 4), cirq.GridQubit(0, 2))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(0, 4), cirq.GridQubit(0, 3))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(0, 5), cirq.GridQubit(0, 4))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(0, 6), cirq.GridQubit(0, 5))]), cirq.Moment([And(cv1=1, cv2=1, uncompute=True).on(cirq.GridQubit(0, 5), cirq.GridQubit(0, 4), cirq.GridQubit(-1, 0))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(0, 6), cirq.GridQubit(0, 4))]), cirq.Moment([cirq.CNOT(cirq.GridQubit(0, 6), cirq.GridQubit(0, 5))])])"
    circuit = eval(hwp_circuit_repr)



    # 2. Use the default conservative model.
    model = conservative_FLASQ_costs

    # 3. Set parameters for the cost model application.
    n_algorithmic_qubits = len(circuit.all_qubits())
    n_total_logical_qubits = n_algorithmic_qubits + 20
    # In this example, we ignore the measurement-limited depth by passing a dummy value.
    dummy_measurement_depth = MeasurementDepth(depth=0)

    # 4. Run the analysis pipeline.
    cbloq, flasqified_circuit = convert_circuit_for_flasq_analysis(circuit)

    flasq_counts = get_cost_value(cbloq, FLASQGateTotals())
    span_info = get_cost_value(cbloq, TotalSpanCost())
    qubit_counts = get_cost_value(cbloq, QubitCount())
    assert qubit_counts == n_algorithmic_qubits

    summary = apply_flasq_cost_model(
        model=model,
        n_total_logical_qubits=n_total_logical_qubits,
        qubit_counts=qubit_counts,
        counts=flasq_counts,
        span_info=span_info,
        measurement_depth=dummy_measurement_depth,
        logical_timesteps_per_measurement=1.0,
        verbosity=2,
    )

    # Resolve symbols: V_CULT_FACTOR, ROTATION_ERROR, and T_REACT.
    summary = summary.resolve_symbols(
        frozendict(
            {V_CULT_FACTOR: 6.0, ROTATION_ERROR: 1e-3, T_REACT: 1.0}
        )
    )

    # 5. Verify the results.

    # This is a regression test against a golden value. Based on my analysis of the
    # latest model, the new expected value is 3156.32.
    assert np.isclose(float(summary.total_spacetime_volume), 3156.32, atol=0.1)


# =============================================================================
# Phase 1: Characterization tests for untested flasq_model branches
# =============================================================================

import warnings as warnings_module

from qualtran.surface_code.flasq.flasq_model import get_rotation_depth
from qualtran.surface_code.flasq.measurement_depth import MeasurementDepth
from qualtran.surface_code.flasq.symbols import MIXED_FALLBACK_T_COUNT, ROTATION_ERROR


class ApplyFlasqCostModelWarningsTestSuite:
    """Characterization tests for warning paths in apply_flasq_cost_model (L446-460)."""

    def test_warns_on_unknown_gate_counts(self):
        """Should warn when counts have bloqs_with_unknown_cost (L446-450)."""
        counts = FLASQGateCounts(
            cnot=10,
            bloqs_with_unknown_cost=frozendict({Hadamard(): 5}),
        )
        span = GateSpan(connect_span=10, compute_span=10)
        md = MeasurementDepth(depth=5)

        with pytest.warns(UserWarning, match="unknown FLASQ counts"):
            apply_flasq_cost_model(
                model=conservative_FLASQ_costs,
                n_total_logical_qubits=50,
                qubit_counts=10,
                counts=counts,
                span_info=span,
                measurement_depth=md,
                logical_timesteps_per_measurement=1.0,
            )

    def test_warns_on_uncounted_spans(self):
        """Should warn when span_info has uncounted_bloqs (L451-455)."""
        counts = FLASQGateCounts(cnot=10)
        span = GateSpan(
            connect_span=10,
            compute_span=10,
            uncounted_bloqs={CNOT(): 3},
        )
        md = MeasurementDepth(depth=5)

        with pytest.warns(UserWarning, match="uncounted span bloqs"):
            apply_flasq_cost_model(
                model=conservative_FLASQ_costs,
                n_total_logical_qubits=50,
                qubit_counts=10,
                counts=counts,
                span_info=span,
                measurement_depth=md,
                logical_timesteps_per_measurement=1.0,
            )

    def test_warns_on_unknown_measurement_depth(self):
        """Should warn when measurement_depth has unknown bloqs (L456-460)."""
        counts = FLASQGateCounts(cnot=10)
        span = GateSpan(connect_span=10, compute_span=10)
        md = MeasurementDepth(
            depth=5,
            bloqs_with_unknown_depth={Hadamard(): 2},
        )

        with pytest.warns(UserWarning, match="unknown measurement depth"):
            apply_flasq_cost_model(
                model=conservative_FLASQ_costs,
                n_total_logical_qubits=50,
                qubit_counts=10,
                counts=counts,
                span_info=span,
                measurement_depth=md,
                logical_timesteps_per_measurement=1.0,
            )

    def test_assumptions_resolve_symbols(self):
        """When assumptions are provided, summary should be resolved (L528-529)."""
        counts = FLASQGateCounts(z_rotation=10)
        span = GateSpan(connect_span=5, compute_span=5)
        md = MeasurementDepth(depth=3)

        assumptions = frozendict({
            ROTATION_ERROR: 1e-3,
            V_CULT_FACTOR: 6.0,
            T_REACT: 1.0,
        })

        summary = apply_flasq_cost_model(
            model=conservative_FLASQ_costs,
            n_total_logical_qubits=50,
            qubit_counts=10,
            counts=counts,
            span_info=span,
            measurement_depth=md,
            logical_timesteps_per_measurement=1.0,
            assumptions=assumptions,
        )
        # Should be fully resolved — no sympy symbols
        assert isinstance(float(summary.total_spacetime_volume), float)
        assert isinstance(float(summary.total_t_count), float)


class GetRotationDepthTestSuite:
    """Characterization tests for get_rotation_depth function."""

    def test_returns_symbolic_without_error(self):
        """Without rotation_error, should return symbolic MIXED_FALLBACK_T_COUNT."""
        result = get_rotation_depth()
        assert result == MIXED_FALLBACK_T_COUNT

    def test_returns_concrete_with_error(self):
        """With rotation_error, should substitute and return a concrete number."""
        result = get_rotation_depth(rotation_error=1e-3)
        assert isinstance(float(result), float)
        assert float(result) > 0

    def test_consistent_with_symbols_test(self):
        """Result should match direct substitution of MIXED_FALLBACK_T_COUNT."""
        import math
        eps = 1e-6
        result = float(get_rotation_depth(rotation_error=eps))
        expected = 4.86 + 0.53 * math.log2(1 / eps)
        assert result == pytest.approx(expected, rel=1e-10)

