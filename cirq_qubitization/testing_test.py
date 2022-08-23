import cirq
import numpy as np
import pytest

from cirq_qubitization import testing as cq_testing
from cirq_qubitization.gate_with_registers import Registers
from cirq_qubitization.testing import get_classical_inputs


def test_assert_circuit_inp_out_cirqsim():
    qubits = cirq.LineQubit.range(4)
    initial_state = [0, 1, 0, 0]
    circuit = cirq.Circuit(cirq.X(qubits[3]))
    final_state = [0, 1, 0, 1]

    cq_testing.assert_circuit_inp_out_cirqsim(circuit, qubits, initial_state, final_state)

    final_state = [0, 0, 0, 1]
    with pytest.raises(AssertionError):
        cq_testing.assert_circuit_inp_out_cirqsim(circuit, qubits, initial_state, final_state)


def test_classical_inputs():
    r = Registers.build(control=2, target=1)
    bitstrings = get_classical_inputs(
        variable_registers=[r['control']], fixed_registers={r['target']: 0}
    )
    should_be = {
        'control': np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        'target': np.array([[0], [0], [0], [0]]),
    }
    assert list(bitstrings) == list(should_be)
    for k in bitstrings:
        np.testing.assert_array_equal(bitstrings[k], should_be[k])
