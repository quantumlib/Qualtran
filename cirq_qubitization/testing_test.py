import cirq
import pytest

from cirq_qubitization import testing as cq_testing


def test_assert_circuit_inp_out_cirqsim():
    qubits = cirq.LineQubit.range(4)
    initial_state = [0, 1, 0, 0]
    circuit = cirq.Circuit(cirq.X(qubits[3]))
    final_state = [0, 1, 0, 1]

    cq_testing.assert_circuit_inp_out_cirqsim(circuit, qubits, initial_state, final_state)

    final_state = [0, 0, 0, 1]
    with pytest.raises(AssertionError):
        cq_testing.assert_circuit_inp_out_cirqsim(circuit, qubits, initial_state, final_state)
