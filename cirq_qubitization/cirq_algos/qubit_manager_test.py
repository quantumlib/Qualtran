import cirq
import pytest

import cirq_qubitization.cirq_algos as cqa


@pytest.mark.parametrize('_', range(2))
def test_simple_qubit_manager(_):
    with cqa.memory_management_context():
        assert cqa.qalloc(1) == [cqa.CleanQubit(0)]
        assert cqa.qalloc(2) == [cqa.CleanQubit(1), cqa.CleanQubit(2)]
        assert cqa.qborrow(1) == [cqa.BorrowableQubit(0)]
        assert cqa.qborrow(2) == [cqa.BorrowableQubit(1), cqa.BorrowableQubit(2)]
        cqa.qfree([cqa.CleanQubit(i) for i in range(3)])
        cqa.qfree([cqa.BorrowableQubit(i) for i in range(3)])
        with pytest.raises(ValueError, match="not allocated"):
            cqa.qfree([cqa.CleanQubit(10)])
        with pytest.raises(ValueError, match="not allocated"):
            cqa.qfree([cqa.BorrowableQubit(10)])


class GateAllocInDecompose(cirq.Gate):
    def __init__(self, num_alloc: int = 1):
        self.num_alloc = num_alloc

    def _num_qubits_(self) -> int:
        return 1

    def _decompose_(self, qubits):
        for q in cqa.qalloc(self.num_alloc):
            yield cirq.CNOT(qubits[0], q)
            cqa.qfree([q])


def test_greedy_qubit_manager():
    def make_circuit():
        q = cirq.LineQubit.range(2)
        g = GateAllocInDecompose(1)
        circuit = cirq.Circuit(cirq.decompose_once(g.on(q[0])), cirq.decompose_once(g.on(q[1])))
        return circuit

    with cqa.memory_management_context(cqa.GreedyQubitManager(prefix="ancilla", size=1)):
        # Qubit manager with only 1 managed qubit. Will always repeat the same qubit.
        circuit = make_circuit()
        cirq.testing.assert_has_diagram(
            circuit,
            """
0: ───────────@───────
              │
1: ───────────┼───@───
              │   │
ancilla_0: ───X───X───
            """,
        )

    with cqa.memory_management_context(cqa.GreedyQubitManager(prefix="ancilla", size=2)):
        # Qubit manager with 2 managed qubits and parallelize=True, tries to minimize adding additional
        # data dependencies by minimizing reuse.
        circuit = make_circuit()
        cirq.testing.assert_has_diagram(
            circuit,
            """
              ┌──┐
0: ────────────@─────
               │
1: ────────────┼@────
               ││
ancilla_0: ────X┼────
                │
ancilla_1: ─────X────
              └──┘
        """,
        )

    with cqa.memory_management_context(
        cqa.GreedyQubitManager(prefix="ancilla", size=2, parallelize=False)
    ):
        # Qubit manager with 2 managed qubits and parallelize=False, tries to minimize reuse by potentially
        # adding new data dependencies.
        circuit = make_circuit()
        cirq.testing.assert_has_diagram(
            circuit,
            """
0: ───────────@───────
              │
1: ───────────┼───@───
              │   │
ancilla_1: ───X───X───
     """,
        )
