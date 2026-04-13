# test_ising_example.py
# Tests for the ising_example.py script.

from typing import Tuple, Optional

import pytest
import cirq
import numpy as np
from frozendict import frozendict
import sympy

# Import functions from the script to be tested
from qualtran.surface_code.flasq.examples.ising import (
    build_ising_circuit,
    ising_zz_layer,
    ising_x_layer,
)

from qualtran.surface_code.flasq.symbols import V_CULT_FACTOR, T_REACT, ROTATION_ERROR

from qualtran.resource_counting import get_cost_value
from qualtran.resource_counting import QubitCount
from qualtran.surface_code.flasq.flasq_model import (
    FLASQCostModel,
    FLASQSummary,
    optimistic_FLASQ_costs,
    apply_flasq_cost_model,
    get_rotation_depth,
)
from qualtran.surface_code.flasq.cirq_interop import convert_circuit_for_flasq_analysis
from qualtran.surface_code.flasq.error_mitigation import calculate_error_mitigation_metrics
from qualtran.surface_code.flasq.measurement_depth import MeasurementDepth, TotalMeasurementDepth
from qualtran.surface_code.flasq.span_counting import GateSpan, TotalSpanCost
from qualtran.surface_code.flasq.utils import substitute_until_fixed_point
from qualtran.surface_code.flasq.volume_counting import (
    FLASQGateCounts,
    FLASQGateTotals,
)


def test_ising_zz_layer_structure():
    """Tests the basic structure of the ZZ layer for a 2x2 lattice."""
    # Setup for a 2x2 lattice
    rows, cols = 2, 2
    qubits = tuple(cirq.GridQubit.rect(rows, cols))  # q(0,0), q(0,1), q(1,0), q(1,1)
    j_coupling = 1.0
    dt_layer = 0.1  # Represents dt/2 in the Trotter step

    ops = list(
        cirq.flatten_op_tree(ising_zz_layer(qubits, rows, cols, j_coupling, dt_layer))
    )

    # --- Analysis for 2x2 lattice ---
    # Total qubits = 4
    # Horizontal loop applies interactions for each qubit with its right neighbor (PBC):
    # q(0,0)<->q(0,1), q(0,1)<->q(0,0), q(1,0)<->q(1,1), q(1,1)<->q(1,0)
    # Total horizontal interactions applied = rows * cols = 4 -> 4 * 3 = 12 gates
    # Vertical loop applies interactions for each qubit with its lower neighbor (PBC):
    # q(0,0)<->q(1,0), q(0,1)<->q(1,1), q(1,0)<->q(0,0), q(1,1)<->q(0,1)
    # Total vertical interactions = rows * cols = 4 -> 4 * 3 = 12 gates
    # Total expected gates = 12 (horizontal) + 12 (vertical) = 24 gates.

    # Check a specific horizontal interaction, e.g., q(0,0)-q(0,1)
    # qubits[0] is q(0,0), qubits[1] is q(0,1)
    rz_angle = 2 * j_coupling * dt_layer  # Rz angle in decomposition
    expected_zz_ops_q00_q01 = [
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.Rz(rads=rz_angle).on(qubits[1]),
        cirq.CNOT(qubits[0], qubits[1]),
    ]

    # Check if these specific ops are present (order might vary slightly depending on loop structure)
    op_strs = {str(op) for op in ops}  # Use a set for faster lookup
    expected_op_strs = [str(op) for op in expected_zz_ops_q00_q01]

    # Check that the horizontal interaction gates are present
    assert expected_op_strs[0] in op_strs
    assert expected_op_strs[1] in op_strs
    assert expected_op_strs[2] in op_strs

    # Check total number of gates based on the implementation logic
    # Number of horizontal pairs = rows * cols
    # Number of vertical pairs = rows * cols
    # Total pairs = 2 * rows * cols
    # Total gates = 3 * Total pairs
    assert len(ops) == 3 * (rows * cols + rows * cols)
    # For 2x2: 3 * (2*2 + 2*2) = 3 * 8 = 24. This matches the analysis.


def test_ising_x_layer_structure():
    """Tests the basic structure of the X layer."""
    qubits = tuple(cirq.GridQubit.rect(2, 2))
    h_field = 0.5
    dt = 0.2

    ops = ising_x_layer(qubits, h_field, dt)

    # Expected: One moment with one Rx gate per qubit
    moment = next(ops)
    assert len(moment.operations) == len(qubits)
    expected_theta = -2 * h_field * dt
    for op in moment.operations:
        assert isinstance(op.gate, cirq.Rx)
        # Use the public 'rads' attribute
        assert np.isclose(op.gate._rads, expected_theta)


def test_build_ising_circuit_basic():
    """Tests if the circuit builder returns a Circuit object for 2x2."""
    # Changed test case to 2x2 to match the zz_layer test update
    rows, cols = 2, 2
    n_steps = 1
    circuit = build_ising_circuit(
        rows=rows, cols=cols, j_coupling=1.0, h_field=0.5, dt=0.1, n_steps=n_steps
    )
    assert isinstance(circuit, cirq.Circuit)

    # --- Analysis for 2x2 lattice ---
    n_qubits = rows * cols  # 4
    # Number of moments from ZZ layer = 12
    # Number of moments from X layers = n_steps + 1
    # Total moments = 12 * n_steps + (n_steps + 1)
    zz_moments = 12  # 2x2 is even x even
    expected_moments = zz_moments * n_steps + (n_steps + 1)

    assert len(circuit.moments) == expected_moments

    # Total ops = (n_steps * 12 moments * (avg ops/moment)) + ((n_steps+1) * 1 moment * (n_qubits ops/moment))
    # Ops per ZZ layer = 2 * rows * cols * 3 = 24
    expected_ops = (n_steps * (rows * cols + rows * cols) * 3) + (
        (n_steps + 1) * n_qubits
    )

    assert len(list(circuit.all_operations())) == expected_ops


def test_build_ising_circuit_basic_odd():
    """Tests if the circuit builder returns a Circuit object for 2x2."""
    rows, cols = 5, 5
    n_steps = 1
    circuit = build_ising_circuit(
        rows=rows, cols=cols, j_coupling=1.0, h_field=0.5, dt=0.1, n_steps=n_steps
    )
    assert isinstance(circuit, cirq.Circuit)

    n_qubits = rows * cols
    # Number of moments from ZZ layer = 12
    zz_moments = 18  # 5x5 is odd x odd
    # Number of moments from X layers = n_steps + 1
    # Total moments = 12 * n_steps + (n_steps + 1)
    expected_moments = zz_moments * n_steps + (n_steps + 1)
    assert len(circuit.moments) == expected_moments

    # Ops per ZZ layer = 2 * rows * cols * 3
    expected_ops = (n_steps * (rows * cols + rows * cols) * 3) + (
        (n_steps + 1) * n_qubits
    )

    assert len(list(circuit.all_operations())) == expected_ops


def test_build_ising_circuit_qubit_count():
    """Tests that the circuit is built on the correct number of qubits."""
    rows, cols = 3, 4
    expected_num_qubits = rows * cols

    circuit = build_ising_circuit(
        rows=rows, cols=cols, j_coupling=1.0, h_field=0.5, dt=0.1, n_steps=1
    )

    circuit_qubits = circuit.all_qubits()
    assert len(circuit_qubits) == expected_num_qubits

    # Optional: Check if they are GridQubits as expected
    assert all(isinstance(q, cirq.GridQubit) for q in circuit_qubits)
    # Optional: Check if the grid dimensions match
    max_row = max(q.row for q in circuit_qubits)
    max_col = max(q.col for q in circuit_qubits)
    assert max_row == rows - 1
    assert max_col == cols - 1


def test_build_ising_circuit_zero_steps():
    """Tests circuit building with zero steps."""
    rows, cols = 2, 2
    expected_num_qubits = rows * cols
    circuit = build_ising_circuit(
        rows=rows, cols=cols, j_coupling=1.0, h_field=0.5, dt=0.1, n_steps=0
    )
    assert isinstance(circuit, cirq.Circuit)
    # Even with zero steps, the circuit should be defined over the qubits
    # (containing Identity gates in the current implementation)
    assert len(circuit.all_qubits()) == expected_num_qubits
    # Check operations count specifically for n_steps=0
    assert (
        len(list(circuit.all_operations())) == expected_num_qubits
    )  # One Identity per qubit


@pytest.mark.parametrize("rows, cols", [(2, 2), (3, 3)])
def test_ising_simulation(rows, cols):
    """Tests building and simulating the circuit for different lattice sizes."""
    j_coupling_strength = 1.0
    h_field_strength = 0.5
    time_step = 0.05  # Smaller step for potentially larger systems
    num_trotter_steps = 2

    print(f"\nTesting simulation for {rows}x{cols} lattice...")
    print(
        f"Parameters: J={j_coupling_strength}, h={h_field_strength}, dt={time_step}, steps={num_trotter_steps}"
    )

    # Build the circuit
    ising_circuit = build_ising_circuit(
        rows=rows,
        cols=cols,
        j_coupling=j_coupling_strength,
        h_field=h_field_strength,
        dt=time_step,
        n_steps=num_trotter_steps,
    )

    print(f"Circuit built with {len(list(ising_circuit.all_operations()))} operations.")
    # Check qubit count here as well
    assert len(ising_circuit.all_qubits()) == rows * cols

    zz_moments = 12
    if cols % 2 != 0:
        zz_moments += 3
    if rows % 2 != 0:
        zz_moments += 3
    expected_moments = num_trotter_steps * zz_moments + (num_trotter_steps + 1)
    assert len(ising_circuit.moments) == expected_moments

    # Total ops = (n_steps * ops_per_zz_layer) + (n_x_layers * ops_per_x_layer)
    expected_ops = (num_trotter_steps * (rows * cols + rows * cols) * 3) + (
        (num_trotter_steps + 1) * rows * cols
    )
    assert len(list(ising_circuit.all_operations())) == expected_ops

    # Simulate the circuit
    simulator = cirq.Simulator()
    num_qubits = rows * cols
    # Ensure initial state matches simulator's expected dtype if necessary
    initial_state = np.zeros(2**num_qubits, dtype=simulator._dtype)
    initial_state[0] = 1.0  # Start in |00...0> state

    try:
        result = simulator.simulate(ising_circuit, initial_state=initial_state)
        final_state_vector = result.final_state_vector
        print(f"Simulation successful for {rows}x{cols}.")
    except Exception as e:
        pytest.fail(f"Simulation failed for {rows}x{cols} lattice: {e}")

    # Basic checks on the result
    assert final_state_vector is not None
    assert final_state_vector.shape == (2**num_qubits,)
    assert np.isclose(np.linalg.norm(final_state_vector), 1.0)  # Check normalization


def test_invalid_inputs():
    """Tests that invalid inputs raise ValueErrors."""
    with pytest.raises(ValueError, match="Lattice dimensions must be positive"):
        build_ising_circuit(rows=0, cols=2, j_coupling=1, h_field=1, dt=0.1, n_steps=1)
    with pytest.raises(ValueError, match="Lattice dimensions must be positive"):
        build_ising_circuit(rows=2, cols=-1, j_coupling=1, h_field=1, dt=0.1, n_steps=1)
    with pytest.raises(ValueError, match="Number of Trotter steps cannot be negative"):
        build_ising_circuit(rows=2, cols=2, j_coupling=1, h_field=1, dt=0.1, n_steps=-1)


def test_build_ising_circuit_invalid_order():
    with pytest.raises(ValueError, match="Trotter order must be 2 or 4"):
        build_ising_circuit(
            rows=2, cols=2, j_coupling=1, h_field=1, dt=0.1, n_steps=1, order=3
        )


def test_both_counts_from_ising_model_circuit():
    rows, cols = 4, 6
    n_steps = 2  # Keep small for test speed
    original_circuit = build_ising_circuit(
        rows=rows, cols=cols, j_coupling=0.8, h_field=1.0, dt=0.03, n_steps=n_steps
    )
    expected_num_ops_original = len(list(original_circuit.all_operations()))

    # Convert using the span-aware function
    cbloq, decomposed_circuit = convert_circuit_for_flasq_analysis(original_circuit)

    # Calculate total span cost
    span_cost_val = get_cost_value(cbloq, TotalSpanCost())

    # Span calculation for Ising model with periodic boundary conditions.
    # Each ZZ interaction is CNOT-Rz-CNOT. CNOTs have dist=1 for adjacent
    # qubits and a larger distance for wrap-around connections.
    # Horizontal span per layer:
    #   There are `rows` rows. Each has `cols-1` CNOT pairs of dist=1 and 1 pair of dist=`cols-1`.
    #   Each ZZ has 2 CNOTs. Total horizontal span = rows * ((cols-1)*1 + 1*(cols-1)) * 2 = 4 * rows * (cols-1)
    # Vertical span per layer:
    #   There are `cols` columns. Each has `rows-1` CNOT pairs of dist=1 and 1 pair of dist=`rows-1`.
    #   Total vertical span = cols * ((rows-1)*1 + 1*(rows-1)) * 2 = 4 * cols * (rows-1)
    span_per_layer = 4 * rows * (cols - 1) + 4 * cols * (rows - 1)
    # The old implementation had a bug where it double-counted interactions.
    # The new one does not.
    # Each ZZ layer has 4 sets of interactions. Each set has 2 CNOT moments.
    # Total span = n_steps * 2 * (span_per_set_1 + ... + span_per_set_4)
    # The factor of 2 comes from the CNOT-Rz-CNOT decomposition.
    # Span per set is sum of distances. For a non-periodic grid, this is simple.
    # For periodic, it's rows * (cols-1) for horizontal + cols * (rows-1) for vertical.
    total_expected_connect_span = (
        2
        * n_steps
        * (
            rows * (cols - 1)
            + rows * (cols - 1)
            + cols * (rows - 1)
            + cols * (rows - 1)
        )
    )
    total_expected_compute_span = total_expected_connect_span

    # Should sum the spans from the BloqWithSpanInfo instances created during conversion
    assert span_cost_val == GateSpan(
        connect_span=total_expected_connect_span,
        compute_span=total_expected_compute_span,
        uncounted_bloqs={},
    )

    # Check the decomposed circuit
    zz_moments = 12
    if cols % 2 != 0:
        zz_moments += 3
    if rows % 2 != 0:
        zz_moments += 3
    assert len(decomposed_circuit.moments) == n_steps * zz_moments + (n_steps + 1)
    # The ising circuit is already decomposed into CNOT, Rz, Rx, so decomposed_circuit should be the same.

    # Calculate FLASQ counts
    flasq_cost_val = get_cost_value(cbloq, FLASQGateTotals())

    # FLASQ counts based on the updated ising_example.py structure
    n_qubits = rows * cols
    n_zz_interactions_per_layer = (
        rows * cols + rows * cols
    )  # horizontal + vertical unique pairs
    n_cnot_per_zz_layer = n_zz_interactions_per_layer * 2  # CNOT-Rz-CNOT
    n_rz_per_zz_layer = n_zz_interactions_per_layer * 1

    total_cnots = n_steps * n_cnot_per_zz_layer
    total_rz = n_steps * n_rz_per_zz_layer
    # X layers: 1 initial dt/2, n_steps-1 full dt, 1 final dt/2 => n_steps+1 layers
    total_rx = (n_steps + 1) * n_qubits

    assert flasq_cost_val == FLASQGateCounts(
        cnot=total_cnots, z_rotation=total_rz, x_rotation=total_rx
    )

    # Example cost calculation with a simple model
    cost_model_concrete = FLASQCostModel(
        cnot_base_volume=3.0,
        rz_clifford_volume=5.0,
        rx_clifford_volume=7.0,
        connect_span_volume=1.0,
        compute_span_volume=1.0,
    )
    total_algo_cost = (
        cost_model_concrete.calculate_volume_required_for_clifford_computation(
            flasq_cost_val, span_cost_val
        )
        + cost_model_concrete.calculate_non_clifford_lattice_surgery_volume(
            flasq_cost_val
        )
    )
    print(
        f"Ising Model ({rows}x{cols}, {n_steps} steps) - Estimated Clifford Volume: {total_algo_cost}"
    )
    # Check if calculation runs without error
    assert (
        total_algo_cost
        == total_cnots * 3.0
        + total_rz * 5.0
        + total_rx * 7.0
        + total_expected_connect_span * 1.0
        + total_expected_compute_span * 1.0
    )

    # Use the default model for symbolic check
    cost_model_default = FLASQCostModel()
    total_algo_clifford_volume = (
        cost_model_default.calculate_volume_required_for_clifford_computation(
            flasq_cost_val, span_cost_val
        )
        + cost_model_default.calculate_non_clifford_lattice_surgery_volume(
            flasq_cost_val
        )
    )

    print(total_algo_clifford_volume)

    expected_algo_clifford_volume = sympy.simplify(
        total_cnots * cost_model_default.cnot_base_volume
        + total_rz * cost_model_default.rz_clifford_volume
        + total_rx * cost_model_default.rx_clifford_volume
        + total_expected_connect_span * cost_model_default.connect_span_volume
        + total_expected_compute_span * cost_model_default.compute_span_volume
    )

    print(expected_algo_clifford_volume)

    assumptions = {ROTATION_ERROR: 1e-3, T_REACT: 1.0}

    # Check against the default costs stored in the model instance
    np.testing.assert_almost_equal(
        substitute_until_fixed_point(
            total_algo_clifford_volume, frozendict(assumptions)
        ),
        substitute_until_fixed_point(
            expected_algo_clifford_volume, frozendict(assumptions)
        ),
    )


# --- Tests for Measurement Depth ---


def test_ising_x_layer_measurement_depth():
    """Tests the measurement depth of a single X layer."""
    rows, cols = 6, 6
    qubits = tuple(cirq.GridQubit.rect(rows, cols))
    h_field = 0.5
    dt = 0.2  # Non-zero dt to ensure non-identity Rx gates

    # The function now returns a generator of moments
    original_x_circuit = cirq.Circuit(ising_x_layer(qubits, h_field, dt))
    cbloq_x_layer, decomposed_x_circuit = convert_circuit_for_flasq_analysis(
        original_x_circuit
    )

    cost_key = TotalMeasurementDepth(rotation_depth=1.0)
    depth_result = get_cost_value(cbloq_x_layer, cost_key)

    # All Rx gates in the layer are parallel, so depth should be 1.0
    assert depth_result == MeasurementDepth(depth=1.0)
    assert len(list(decomposed_x_circuit.all_operations())) == rows * cols


def test_ising_zz_layer_measurement_depth():
    """Tests the measurement depth of a single ZZ layer."""
    rows, cols = 6, 6
    qubits = tuple(cirq.GridQubit.rect(rows, cols))
    j_coupling = 1.0
    dt_layer = 0.1

    original_zz_circuit = cirq.Circuit(
        ising_zz_layer(qubits, rows, cols, j_coupling, dt_layer)
    )
    cbloq_zz_layer, decomposed_zz_circuit = convert_circuit_for_flasq_analysis(
        original_zz_circuit
    )

    cost_key = TotalMeasurementDepth(rotation_depth=1.0)
    depth_result = get_cost_value(cbloq_zz_layer, cost_key)

    # The depth coster only counts non-Clifford gates by default. CNOTs are Clifford.
    # The Rz gates are non-Clifford. There are 4 layers of them.
    assert depth_result == MeasurementDepth(depth=4.0)
    # Total ops = (rows * cols + rows * cols) * 3
    assert (
        len(list(decomposed_zz_circuit.all_operations()))
        == (rows * cols + rows * cols) * 3
    )


@pytest.mark.parametrize(
    "n_steps, dt_param, expected_depth_val",
    [
        (1, 0.1, 6.0),  # X(1) + ZZ(4) + X(1) = 6
        (2, 0.1, 11.0),  # X(1) + ZZ(4) + X(1) + ZZ(4) + X(1) = 11
    ],
)
def test_full_ising_circuit_measurement_depth(n_steps, dt_param, expected_depth_val):
    """Tests the measurement depth of the full Ising circuit."""
    rows, cols = 4, 4
    j_coupling = 1.0
    h_field = 0.5

    # For n_steps = 0, if dt_param is 0, Rx angles will be 0, making them identities.
    # TotalMeasurementDepth should recognize Rx(0) as Clifford and assign depth 0.
    original_circuit = build_ising_circuit(
        rows=rows,
        cols=cols,
        j_coupling=j_coupling,
        h_field=h_field,
        dt=dt_param,
        n_steps=n_steps,
    )
    cbloq_full_circuit, decomposed_full_circuit = convert_circuit_for_flasq_analysis(
        original_circuit
    )
    expected_ops_full = len(list(original_circuit.all_operations()))

    cost_key = TotalMeasurementDepth(rotation_depth=1.0)
    depth_result = get_cost_value(cbloq_full_circuit, cost_key)

    if n_steps == 0 and dt_param == 0.0:
        # Rx(0) gates are identities, depth 0
        assert depth_result == MeasurementDepth(depth=0.0)
    elif n_steps == 0 and dt_param != 0.0:
        # One X layer with dt/2
        assert depth_result == MeasurementDepth(depth=1.0)
    else:
        # General case: (n_steps + 1) X-layers and n_steps ZZ-layers
        # Depth of X layer = 1.0, Depth of ZZ layer = 4.0 (from Rz gates)
        calculated_expected_depth = (n_steps + 1) * 1.0 + n_steps * 4.0
        assert depth_result == MeasurementDepth(depth=calculated_expected_depth)
        assert (
            calculated_expected_depth == expected_depth_val
        )  # Sanity check parameterization
    assert len(list(decomposed_full_circuit.all_operations())) == expected_ops_full


def _find_min_time_config_and_summary(
    rows: int,
    cols: int,
    n_steps: int,
    total_allowable_rotation_error: float,
    cultivation_error_rate: float,
    phys_error_rate: float,
    n_total_physical_qubits_available: int,
    time_per_surface_code_cycle: float,
) -> Tuple[float, FLASQSummary]:
    """
    Helper function to find the configuration (code_distance) that yields the
    minimum effective_time_per_noiseless_sample for an Ising model simulation.

    Returns:
        A tuple containing:
            - The minimum effective_time_per_noiseless_sample found.
            - The FLASQSummary object corresponding to this minimum time.
    """
    original_circuit = build_ising_circuit(
        rows=rows, cols=cols, j_coupling=1, h_field=3.04438, dt=0.04, n_steps=n_steps
    )
    cbloq, _ = convert_circuit_for_flasq_analysis(
        original_circuit
    )  # decomposed_circuit not used here

    flasq_counts = get_cost_value(cbloq, FLASQGateTotals())
    total_span = get_cost_value(cbloq, TotalSpanCost())
    qubit_counts = get_cost_value(cbloq, QubitCount())

    if flasq_counts.total_rotations == 0:
        # Avoid division by zero if there are no rotations
        individual_allowable_rotation_error = total_allowable_rotation_error
        # If no rotations, rotation_depth doesn't strictly matter but set to 0 for clarity
        rotation_depth_val = 0.0
    else:
        individual_allowable_rotation_error = (
            total_allowable_rotation_error / flasq_counts.total_rotations
        )
        rotation_depth_val = get_rotation_depth(
            rotation_error=individual_allowable_rotation_error
        )

    measurement_depth = get_cost_value(
        cbloq, TotalMeasurementDepth(rotation_depth=rotation_depth_val)
    )

    min_effective_time = np.inf
    lambda_val = 1e-2 / phys_error_rate

    summary_for_min_time: Optional[FLASQSummary] = None

    for code_distance in range(5, 50, 2):
        logical_timestep_per_measurement = 10 / code_distance
        # Ensure denominator is not zero and result is int
        denominator = 2 * (code_distance + 1) ** 2
        if denominator == 0:
            continue
        # Calculate the number of logical qubits that can be supported
        n_total_logical_qubits = n_total_physical_qubits_available // denominator
        assert qubit_counts == rows * cols

        if n_total_logical_qubits - qubit_counts > 0:
            flasq_summary = apply_flasq_cost_model(
                model=optimistic_FLASQ_costs,
                n_total_logical_qubits=n_total_logical_qubits,
                qubit_counts=qubit_counts,
                counts=flasq_counts,
                span_info=total_span,
                measurement_depth=measurement_depth,
                logical_timesteps_per_measurement=logical_timestep_per_measurement,
            )

            flasq_summary_resolved = flasq_summary.resolve_symbols(
                frozendict(
                    {
                        ROTATION_ERROR: individual_allowable_rotation_error,
                        V_CULT_FACTOR: 6.0,
                        T_REACT: 1.0,
                    }
                )
            )

            # calculate_error_mitigation_metrics returns (effective_time, wall_clock_time)
            # as error_rate is commented out in its return statement.
            effective_time, _, _ = calculate_error_mitigation_metrics(
                flasq_summary=flasq_summary_resolved,
                time_per_surface_code_cycle=time_per_surface_code_cycle,
                code_distance=code_distance,
                lambda_val=lambda_val,
                cultivation_error_rate=cultivation_error_rate,
            )
            if effective_time < min_effective_time:
                min_effective_time = effective_time
                summary_for_min_time = flasq_summary_resolved

    if summary_for_min_time is None:
        raise ValueError(
            f"Could not find a valid configuration for {rows}x{cols} Ising model."
        )

    return min_effective_time, summary_for_min_time


def test_ising_volume_limited_depth_comparison_5x5_vs_6x6():
    """
    Compares the volume_limited_depth for 5x5 vs 6x6 Ising models.

    The volume_limited_depth is taken from the FLASQ summary that corresponds
    to the configuration (i.e., code_distance) achieving the minimum
    effective_time_per_noiseless_sample for each model size, under specific
    optimistic FLASQ parameters.

    The test asserts that the volume_limited_depth for the 5x5 case is
    strictly less than that of the 6x6 case. This should always be smaller, although
    the time to solution may be larger for the 5x5 case due to measurement depth
    constraints.
    """
    # Parameters from the notebook
    n_steps = 18
    total_allowable_rotation_error = 0.005
    cultivation_error_rate = 1e-8
    phys_error_rate = 1e-4  # Specific physical error rate for this test
    n_total_physical_qubits_available = 10000
    time_per_surface_code_cycle = 1e-6

    # Find the configuration that minimizes effective time for 5x5, and get its summary
    min_time_5x5, summary_5x5 = _find_min_time_config_and_summary(
        rows=5,
        cols=5,
        n_steps=n_steps,
        total_allowable_rotation_error=total_allowable_rotation_error,
        cultivation_error_rate=cultivation_error_rate,
        phys_error_rate=phys_error_rate,
        n_total_physical_qubits_available=n_total_physical_qubits_available,
        time_per_surface_code_cycle=time_per_surface_code_cycle,
    )
    print(
        f"For 5x5: Min effective time = {min_time_5x5}, Volume-Limited Depth = {summary_5x5.volume_limited_depth}"
    )

    # Find the configuration that minimizes effective time for 6x6, and get its summary
    min_time_6x6, summary_6x6 = _find_min_time_config_and_summary(
        rows=6,
        cols=6,
        n_steps=n_steps,
        total_allowable_rotation_error=total_allowable_rotation_error,
        cultivation_error_rate=cultivation_error_rate,
        phys_error_rate=phys_error_rate,
        n_total_physical_qubits_available=n_total_physical_qubits_available,
        time_per_surface_code_cycle=time_per_surface_code_cycle,
    )
    print(
        f"For 6x6: Min effective time = {min_time_6x6}, Volume-Limited Depth = {summary_6x6.volume_limited_depth}"
    )

    # Assert that the volume_limited_depth for the 5x5 case is less than that of the 6x6 case.
    assert summary_5x5.volume_limited_depth < summary_6x6.volume_limited_depth, (
        f"5x5 volume_limited_depth ({summary_5x5.volume_limited_depth}) "
        f"should be < 6x6 volume_limited_depth ({summary_6x6.volume_limited_depth})"
    )


@pytest.mark.parametrize(
    "rows, cols", [(4, 4), (5, 5), (5, 4), (4, 5), (7, 7), (8, 8), (10, 10)]
)
def test_ising_zz_layer_moment_structure(rows, cols):
    """Phase 1: Verify that the ZZ layer is parallelized correctly.

    The `ising_zz_layer` function generates operations in a checkerboard-like
    pattern. When these are appended to a circuit with `InsertStrategy.NEW`,
    Cirq's moment-packing algorithm should place non-conflicting gates into
    the same moment.

    For a 2D lattice with periodic boundary conditions, the ZZ interactions
    can be grouped into 4 parallel sets (horizontal-even, horizontal-odd,
    vertical-even, vertical-odd). Each set is decomposed into CNOT-Rz-CNOT,
    which takes 3 moments. For odd dimensions, boundary interactions must be
    separated, adding 3 moments per odd dimension.
    - even x even: 12 moments
    - odd x even / even x odd: 15 moments
    - odd x odd: 18 moments
    """
    qubits = tuple(cirq.GridQubit.rect(rows, cols))
    j_coupling = 1.0
    dt_layer = 0.1

    # 1. Construct a circuit from the generator of moments.
    circuit = cirq.Circuit(ising_zz_layer(qubits, rows, cols, j_coupling, dt_layer))

    # 3. Assert that the depth is correct based on lattice dimensions.
    expected_depth = 12
    if rows % 2 != 0:
        expected_depth += 3
    if cols % 2 != 0:
        expected_depth += 3

    assert len(circuit.moments) == expected_depth


def test_build_ising_circuit_4th_order_gate_counts():
    """Tests gate counts for the 4th-order Trotter implementation."""
    rows, cols = 10, 10
    n_steps = 20
    n_qubits = rows * cols

    circuit = build_ising_circuit(
        rows=rows,
        cols=cols,
        j_coupling=1.0,
        h_field=0.5,
        dt=0.1,
        n_steps=n_steps,
        order=4,
    )

    # Expected counts for 10x10 (N=100), T=20 steps:
    # Total ZZ interactions: 2*N per U2 step * 5 U2 steps per U4 step * T U4 steps
    # = 2 * n_qubits * 5 * n_steps
    total_zz_interactions = 2 * n_qubits * 5 * n_steps
    expected_cnots = total_zz_interactions * 2
    expected_rz = total_zz_interactions

    # Total Rx layers: 5*T U2 steps means 5*T+1 X-layers
    expected_rx = n_qubits * (5 * n_steps + 1)

    # Verify individual gate types by iterating through all operations
    cnot_count = 0
    rz_count = 0
    rx_count = 0
    for op in circuit.all_operations():
        if op.gate == cirq.CNOT:
            cnot_count += 1
        elif isinstance(op.gate, cirq.Rz):
            rz_count += 1
        elif isinstance(op.gate, cirq.Rx):
            rx_count += 1

    assert cnot_count == expected_cnots
    assert rz_count == expected_rz
    assert rx_count == expected_rx


def test_build_ising_circuit_4th_order_measurement_depth():
    """Tests measurement depth for the 4th-order Trotter implementation."""
    rows, cols = 4, 4
    n_steps = 2
    circuit = build_ising_circuit(
        rows=rows,
        cols=cols,
        j_coupling=1.0,
        h_field=0.5,
        dt=0.1,
        n_steps=n_steps,
        order=4,
    )
    cbloq, _ = convert_circuit_for_flasq_analysis(circuit)
    depth_result = get_cost_value(cbloq, TotalMeasurementDepth(rotation_depth=1.0))
    expected_depth = (5 * n_steps + 1) * 1.0 + (5 * n_steps) * 4.0
    assert depth_result.depth == expected_depth


# --- Tests for Open Boundary Conditions ---


@pytest.mark.parametrize("rows, cols", [(4, 4), (5, 5), (4, 5)])
def test_ising_zz_layer_open_boundary_counts(rows, cols):
    """Tests that the correct number of interactions are generated for open boundaries."""
    qubits = tuple(cirq.GridQubit.rect(rows, cols))
    j_coupling = 1.0
    dt_layer = 0.1

    circuit = cirq.Circuit(
        ising_zz_layer(
            qubits, rows, cols, j_coupling, dt_layer, periodic_boundary=False
        )
    )
    ops = list(circuit.all_operations())

    # For an open boundary grid:
    # - Horizontal interactions: rows * (cols - 1)
    # - Vertical interactions: cols * (rows - 1)
    # Each interaction is CNOT-Rz-CNOT, so 3 ops total.
    expected_h_interactions = rows * (cols - 1)
    expected_v_interactions = cols * (rows - 1)
    expected_total_ops = 3 * (expected_h_interactions + expected_v_interactions)

    assert len(ops) == expected_total_ops


@pytest.mark.parametrize("rows, cols", [(4, 4), (5, 5)])
def test_build_ising_circuit_open_boundary_counts(rows, cols):
    """Tests the full circuit builder with open boundary conditions."""
    n_steps = 2
    circuit = build_ising_circuit(
        rows=rows,
        cols=cols,
        j_coupling=1.0,
        h_field=0.5,
        dt=0.1,
        n_steps=n_steps,
        periodic_boundary=False,
    )

    # Expected ops for one ZZ layer with open boundaries
    h_interactions = rows * (cols - 1)
    v_interactions = cols * (rows - 1)
    ops_per_zz_layer = 3 * (h_interactions + v_interactions)

    # Expected ops for X layers
    ops_per_x_layer = rows * cols
    total_x_ops = (n_steps + 1) * ops_per_x_layer

    assert (
        len(list(circuit.all_operations())) == n_steps * ops_per_zz_layer + total_x_ops
    )
