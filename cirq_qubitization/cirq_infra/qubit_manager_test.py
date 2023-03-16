import cirq
import pytest

import cirq_qubitization.cirq_infra as cqi


@pytest.mark.parametrize('_', range(2))
def test_simple_qubit_manager(_):
    with cqi.memory_management_context():
        assert cqi.qalloc(1) == [cqi.CleanQubit(0)]
        assert cqi.qalloc(2) == [cqi.CleanQubit(1), cqi.CleanQubit(2)]
        assert cqi.qborrow(1) == [cqi.BorrowableQubit(0)]
        assert cqi.qborrow(2) == [cqi.BorrowableQubit(1), cqi.BorrowableQubit(2)]
        cqi.qfree([cqi.CleanQubit(i) for i in range(3)])
        cqi.qfree([cqi.BorrowableQubit(i) for i in range(3)])
        with pytest.raises(ValueError, match="not allocated"):
            cqi.qfree([cqi.CleanQubit(10)])
        with pytest.raises(ValueError, match="not allocated"):
            cqi.qfree([cqi.BorrowableQubit(10)])


class GateAllocInDecompose(cirq.Gate):
    def __init__(self, num_alloc: int = 1):
        self.num_alloc = num_alloc

    def _num_qubits_(self) -> int:
        return 1

    def _decompose_(self, qubits):
        for q in cqi.qalloc(self.num_alloc):
            yield cirq.CNOT(qubits[0], q)
            cqi.qfree([q])

    def __str__(self):
        return 'TestGateAlloc'


def test_greedy_qubit_manager():
    def make_circuit():
        q = cirq.LineQubit.range(2)
        g = GateAllocInDecompose(1)
        circuit = cirq.Circuit(cirq.decompose_once(g.on(q[0])), cirq.decompose_once(g.on(q[1])))
        return circuit

    with cqi.memory_management_context(cqi.GreedyQubitManager(prefix="ancilla", size=1)):
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

    with cqi.memory_management_context(cqi.GreedyQubitManager(prefix="ancilla", size=2)):
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

    with cqi.memory_management_context(
        cqi.GreedyQubitManager(prefix="ancilla", size=2, parallelize=False)
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
