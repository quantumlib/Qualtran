import cirq
import numpy as np
import pytest

import cirq_qubitization
from cirq_qubitization import testing as cq_testing


@pytest.mark.parametrize("n", [*range(3, 20), 25, 41])
@pytest.mark.parametrize("num_controls", [0, 1])
def test_prepare_uniform_superposition(n, num_controls):
    gate = cirq_qubitization.PrepareUniformSuperposition(n, num_controls=num_controls)
    all_qubits = cirq.LineQubit.range(cirq.num_qubits(gate))
    control, target = all_qubits[:num_controls], all_qubits[num_controls:]
    prepare_uniform_op = gate.on(*control, *target)
    circuit = cirq.Circuit(cirq.decompose(prepare_uniform_op))
    all_qubits = sorted(circuit.all_qubits())
    qubit_vals = {x: 0 for x in all_qubits}
    qubit_vals.update({x: 1 for x in control})
    initial_state = [qubit_vals[x] for x in all_qubits]
    result = cirq.Simulator(dtype=np.complex128).simulate(
        circuit, initial_state=initial_state, qubit_order=all_qubits
    )
    final_target_state = cirq.sub_state_vector(
        result.final_state_vector, keep_indices=[all_qubits.index(q) for q in target]
    )
    expected_target_state = np.asarray([np.sqrt(1.0 / n)] * n + [0] * (2 ** len(target) - n))
    cirq.testing.assert_allclose_up_to_global_phase(
        expected_target_state, final_target_state, atol=1e-6
    )
