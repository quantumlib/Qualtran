import cirq
import pytest
from cirq_qubitization.t_complexity_protocol import TComplexity

import cirq_qubitization.testing as cq_testing
from cirq_qubitization.and_gate import And
from cirq_qubitization.gate_with_registers import Registers


def test_assert_circuit_inp_out_cirqsim():
    qubits = cirq.LineQubit.range(4)
    initial_state = [0, 1, 0, 0]
    circuit = cirq.Circuit(cirq.X(qubits[3]))
    final_state = [0, 1, 0, 1]

    cq_testing.assert_circuit_inp_out_cirqsim(circuit, qubits, initial_state, final_state)

    final_state = [0, 0, 0, 1]
    with pytest.raises(AssertionError):
        cq_testing.assert_circuit_inp_out_cirqsim(circuit, qubits, initial_state, final_state)


def test_gate_helper():
    g = cq_testing.GateHelper(And(cv=(1, 0, 1, 0)))
    assert g.gate == And(cv=(1, 0, 1, 0))
    assert g.r == Registers.build(control=4, ancilla=2, target=1)
    assert g.quregs == {
        'control': cirq.NamedQubit.range(4, prefix='control'),
        'ancilla': cirq.NamedQubit.range(2, prefix='ancilla'),
        'target': [cirq.NamedQubit('target')],
    }
    assert g.operation.qubits == tuple(g.all_qubits)
    assert len(g.circuit) == 1


class DoesNotDecompose(cirq.Operation):
    def _t_complexity_(self) -> TComplexity:
        return TComplexity(t=1, clifford=2, rotations=3)

    @property
    def qubits(self):
        return []

    def with_qubits(self, _):
        pass


class InconsistentDecompostion(cirq.Operation):
    def _t_complexity_(self) -> TComplexity:
        return TComplexity(rotations=1)

    def _decompose_(self) -> cirq.OP_TREE:
        yield cirq.X(self.qubits[0])

    @property
    def qubits(self):
        return tuple(cirq.LineQubit(3).range(3))

    def with_qubits(self, _):
        pass


@pytest.mark.parametrize(
    "val", [cirq.T, DoesNotDecompose(), cq_testing.GateHelper(And()).operation]
)
def test_assert_decompose_is_consistent_with_t_complexity(val):
    cq_testing.assert_decompose_is_consistent_with_t_complexity(val)


def test_assert_decompose_is_consistent_with_t_complexity_raises():
    with pytest.raises(AssertionError):
        cq_testing.assert_decompose_is_consistent_with_t_complexity(InconsistentDecompostion())
