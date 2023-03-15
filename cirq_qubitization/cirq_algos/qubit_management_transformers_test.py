import cirq

import cirq_qubitization.cirq_algos as cqa


class GateAllocAndBorrowInDecompose(cirq.Gate):
    def __init__(self, num_alloc: int = 1):
        self.num_alloc = num_alloc

    def _num_qubits_(self) -> int:
        return 1

    def __str__(self) -> str:
        return 'TestGate'

    def _decompose_(self, qubits):
        qa, qb = cqa.qalloc(self.num_alloc), cqa.qborrow(self.num_alloc)
        for q, b in zip(qa, qb):
            yield cirq.CSWAP(qubits[0], q, b)
        yield cirq.qft(*qb).controlled_by(qubits[0])
        for q, b in zip(qa, qb):
            yield cirq.CSWAP(qubits[0], q, b)
        cqa.qfree(qa + qb)


def test_decompose_and_allocate_qubits():
    with cqa.memory_management_context():
        op = GateAllocAndBorrowInDecompose(3).on(cirq.NamedQubit("original"))
        extra = cirq.LineQubit.range(3)
        circuit = cirq.Circuit(cirq.H.on_each(*extra), cirq.Moment(op), cirq.decompose_once(op))
        cirq.testing.assert_has_diagram(
            circuit,
            """
_b0: ───────────────────────×───────────qft───×───────────
                            │           │     │
_b1: ───────────────────────┼───×───────#2────┼───×───────
                            │   │       │     │   │
_b2: ───────────────────────┼───┼───×───#3────┼───┼───×───
                            │   │   │   │     │   │   │
_c0: ───────────────────────×───┼───┼───┼─────×───┼───┼───
                            │   │   │   │     │   │   │
_c1: ───────────────────────┼───×───┼───┼─────┼───×───┼───
                            │   │   │   │     │   │   │
_c2: ───────────────────────┼───┼───×───┼─────┼───┼───×───
                            │   │   │   │     │   │   │
0: ──────────H──────────────┼───┼───┼───┼─────┼───┼───┼───
                            │   │   │   │     │   │   │
1: ──────────H──────────────┼───┼───┼───┼─────┼───┼───┼───
                            │   │   │   │     │   │   │
2: ──────────H──────────────┼───┼───┼───┼─────┼───┼───┼───
                            │   │   │   │     │   │   │
original: ───────TestGate───@───@───@───@─────@───@───@───
             """,
        )
        allocated_circuit = cqa.decompose_and_allocate_qubits(circuit, decompose=lambda op: op)
        cirq.testing.assert_has_diagram(
            allocated_circuit,
            """
0: ───────────H──────────────×───────────qft───×───────────
                             │           │     │
1: ───────────H──────────────┼───×───────#2────┼───×───────
                             │   │       │     │   │
2: ───────────H──────────────┼───┼───×───#3────┼───┼───×───
                             │   │   │   │     │   │   │
ancilla_0: ──────────────────×───┼───┼───┼─────×───┼───┼───
                             │   │   │   │     │   │   │
ancilla_1: ──────────────────┼───×───┼───┼─────┼───×───┼───
                             │   │   │   │     │   │   │
ancilla_2: ──────────────────┼───┼───×───┼─────┼───┼───×───
                             │   │   │   │     │   │   │
original: ────────TestGate───@───@───@───@─────@───@───@───
""",
        )

        def decompose_func(op):
            return (
                cirq.decompose_once(op)
                if isinstance(op.gate, GateAllocAndBorrowInDecompose)
                else op
            )

        allocated_and_decomposed_circuit = cqa.decompose_and_allocate_qubits(
            circuit, decompose=decompose_func
        )
        cirq.testing.assert_has_diagram(
            allocated_and_decomposed_circuit,
            """
0: ───────────H───×───────────qft───×───────────×───────────qft───×───────────
                  │           │     │           │           │     │
1: ───────────H───┼───×───────#2────┼───×───────┼───×───────#2────┼───×───────
                  │   │       │     │   │       │   │       │     │   │
2: ───────────H───┼───┼───×───#3────┼───┼───×───┼───┼───×───#3────┼───┼───×───
                  │   │   │   │     │   │   │   │   │   │   │     │   │   │
ancilla_0: ───────×───┼───┼───┼─────×───┼───┼───×───┼───┼───┼─────×───┼───┼───
                  │   │   │   │     │   │   │   │   │   │   │     │   │   │
ancilla_1: ───────┼───×───┼───┼─────┼───×───┼───┼───×───┼───┼─────┼───×───┼───
                  │   │   │   │     │   │   │   │   │   │   │     │   │   │
ancilla_2: ───────┼───┼───×───┼─────┼───┼───×───┼───┼───×───┼─────┼───┼───×───
                  │   │   │   │     │   │   │   │   │   │   │     │   │   │
original: ────────@───@───@───@─────@───@───@───@───@───@───@─────@───@───@───
            """,
        )

        # Since we preserve moment structure, if TestGate is in the first moment then
        # we end up allocating 6 ancilla to not introduce any additional data dependency
        # due to our allocation strategy and thus not impact the circuit depth.
        allocated_and_decomposed_circuit = cqa.decompose_and_allocate_qubits(
            cirq.align_left(circuit), decompose=decompose_func
        )
        cirq.testing.assert_has_diagram(
            allocated_and_decomposed_circuit,
            """
0: ───────────H─────────────────────────────×───────────qft───×───────────
                                            │           │     │
1: ───────────H─────────────────────────────┼───×───────#2────┼───×───────
                                            │   │       │     │   │
2: ───────────H─────────────────────────────┼───┼───×───#3────┼───┼───×───
                                            │   │   │   │     │   │   │
ancilla_0: ───×───────────qft───×───────────×───┼───┼───┼─────×───┼───┼───
              │           │     │           │   │   │   │     │   │   │
ancilla_1: ───┼───×───────#2────┼───×───────┼───×───┼───┼─────┼───×───┼───
              │   │       │     │   │       │   │   │   │     │   │   │
ancilla_2: ───┼───┼───×───#3────┼───┼───×───┼───┼───×───┼─────┼───┼───×───
              │   │   │   │     │   │   │   │   │   │   │     │   │   │
ancilla_3: ───×───┼───┼───┼─────×───┼───┼───┼───┼───┼───┼─────┼───┼───┼───
              │   │   │   │     │   │   │   │   │   │   │     │   │   │
ancilla_4: ───┼───×───┼───┼─────┼───×───┼───┼───┼───┼───┼─────┼───┼───┼───
              │   │   │   │     │   │   │   │   │   │   │     │   │   │
ancilla_5: ───┼───┼───×───┼─────┼───┼───×───┼───┼───┼───┼─────┼───┼───┼───
              │   │   │   │     │   │   │   │   │   │   │     │   │   │
original: ────@───@───@───@─────@───@───@───@───@───@───@─────@───@───@───
            """,
        )
