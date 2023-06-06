import cirq
import numpy as np
import pytest

import cirq_qubitization


@pytest.mark.parametrize("n", [*range(3, 20), 25, 41])
@pytest.mark.parametrize("num_controls", [0, 1])
def test_prepare_uniform_superposition(n, num_controls):
    gate = cirq_qubitization.PrepareUniformSuperposition(n, cvs=[1] * num_controls)
    all_qubits = cirq.LineQubit.range(cirq.num_qubits(gate))
    control, target = (all_qubits[:num_controls], all_qubits[num_controls:])
    turn_on_controls = [cirq.X(c) for c in control]
    prepare_uniform_op = gate.on(*control, *target)
    circuit = cirq.Circuit(turn_on_controls, prepare_uniform_op)
    result = cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=all_qubits)
    final_target_state = cirq.sub_state_vector(
        result.final_state_vector,
        keep_indices=list(range(num_controls, num_controls + len(target))),
    )
    expected_target_state = np.asarray([np.sqrt(1.0 / n)] * n + [0] * (2 ** len(target) - n))
    cirq.testing.assert_allclose_up_to_global_phase(
        expected_target_state, final_target_state, atol=1e-6
    )


@pytest.mark.parametrize("n", [*range(3, 41, 3)])
def test_prepare_uniform_superposition_t_complexity(n: int):
    gate = cirq_qubitization.PrepareUniformSuperposition(n)
    result = cirq_qubitization.t_complexity(gate)
    assert result.rotations <= 2
    # TODO(#235): Uncomputing `LessThanGate` should take 0 T-gates instead of 4 * n
    # and therefore the total complexity should come down to `8 * logN`
    assert result.t <= 12 * (n - 1).bit_length()

    gate = cirq_qubitization.PrepareUniformSuperposition(n, cvs=(1,))
    result = cirq_qubitization.t_complexity(gate)
    # TODO(#233): Controlled-H is currently counted as a separate rotation, but it can be
    # implemented using 2 T-gates.
    assert result.rotations <= 2 + 2 * gate.registers.bitsize
    assert result.t <= 12 * (n - 1).bit_length()
