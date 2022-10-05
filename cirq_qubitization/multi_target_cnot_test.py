import pytest
import cirq
import cirq_qubitization
import numpy as np

from cirq_qubitization.testing import assert_decompose_is_consistent_with_t_complexity

@pytest.mark.parametrize("num_targets", [3, 4, 6, 8, 10])
def test_multi_target_cnot(num_targets):
    qubits = cirq.LineQubit.range(num_targets + 1)
    naive_circuit = cirq.Circuit(cirq.CNOT(qubits[0], q) for q in qubits[1:])
    op = cirq_qubitization.MultiTargetCNOT(num_targets).on(*qubits)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        cirq.Circuit(op), naive_circuit, atol=1e-6
    )
    optimal_circuit = cirq.Circuit(cirq.decompose_once(op))
    assert len(optimal_circuit) == 2 * np.ceil(np.log2(num_targets)) + 1


def test_t_complexity():
    for n in range(1, 5 + 1):
        g = cirq_qubitization.MultiTargetCNOT(n)
        assert_decompose_is_consistent_with_t_complexity(g, check_clifford=True)

