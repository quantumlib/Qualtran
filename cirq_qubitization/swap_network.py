from typing import Sequence
import cirq
from cirq_qubitization import multi_target_cnot


def swap_n(control: cirq.Qid, q_x: Sequence[cirq.Qid], q_y: Sequence[cirq.Qid]):
    """Approximately implements a multi-target controlled swap unitary using only 4 * N T-gates.

    Implements the unitary $CSWAP_{n} = |0><0| I + |1><1| SWAP_{n}$ such that the output state is
    correct up to a global phase factor of +1 / -1.

    This is useful when the incorrect phase can be absorbed in a garbage state of an algorithm; and
    thus ignored. See Appendix B.2.c of https://arxiv.org/abs/1812.00954 for more details.
    """

    def g(q: cirq.Qid, adjoint=False) -> cirq.OP_TREE:
        yield [cirq.S(q), cirq.H(q)]
        yield cirq.T(q) ** (1 - 2 * adjoint)
        yield [cirq.H(q), cirq.S(q) ** -1]

    assert len(q_x) == len(q_y), "Registers to swap must be of the same length."

    cnot_x_to_y = [cirq.CNOT(x, y) for x, y in zip(q_x, q_y)]
    cnot_y_to_x = [cirq.CNOT(y, x) for x, y in zip(q_x, q_y)]
    g_inv_on_y = [list(g(q, True)) for q in q_y]  # Uses len(q_y) T-gates
    g_on_y = [list(g(q)) for q in q_y]  # Uses len(q_y) T-gates

    yield [cnot_y_to_x, g_inv_on_y, cnot_x_to_y, g_inv_on_y]
    yield multi_target_cnot.MultiTargetCNOT(len(q_y)).on(control, *q_y)
    yield [g_on_y, cnot_x_to_y, g_on_y, cnot_y_to_x]


class SwapWithZeroGate(cirq.Gate):
    """Swaps |Psi_0> with |Psi_x> if selection register stores index `x`.

    Implements the unitary U |x> |Psi_0> |Psi_1> ... |Psi_n> --> |x> |Psi_x> |Rest of Psi>.
    Note that the state of `|Rest of Psi>` is allowed to be anything and should not be depended
    upon.

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis](https://arxiv.org/abs/1812.00954).
            Low, Kliuchnikov, Schaeffer. 2018.
    """

    def __init__(
        self,
        selection_register: int,
        target_register_bit_size: int,
        target_register_length: int,
    ):
        assert target_register_length <= 2**selection_register
        self.selection_register = selection_register
        self.target_register_bit_size = target_register_bit_size
        self.target_register_length = target_register_length

    def _num_qubits_(self) -> int:
        return (
            self.selection_register
            + self.target_register_length * self.target_register_bit_size
        )

    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        selection = qubits[: self.selection_register]
        target = [
            qubits[st : st + self.target_register_bit_size]
            for st in range(
                self.selection_register, len(qubits), self.target_register_bit_size
            )
        ]
        assert len(target) == self.target_register_length
        for j in range(len(selection)):
            for i in range(len(target) - 2**j):
                yield swap_n(
                    selection[len(selection) - j - 1], target[i], target[i + 2**j]
                )

    def on_registers(
        self, *, selection: Sequence[cirq.Qid], target: Sequence[Sequence[cirq.Qid]]
    ) -> cirq.GateOperation:
        assert len(selection) == self.selection_register
        assert len(target) == self.target_register_length
        assert all(len(t) == self.target_register_bit_size for t in target)
        flat_target = [q for t in target for q in t]
        return cirq.GateOperation(self, selection + flat_target)

    def __repr__(self) -> str:
        return (
            "cirq_qubitization.SwapWithZeroGate("
            f"{self.selection_register},"
            f"{self.target_register_bit_size},"
            f"{self.target_register_length}"
            f")"
        )

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["@(n)"] * self.selection_register
        wire_symbols += ["swap_0"] * self.target_register_bit_size
        wire_symbols += (
            ["swap_r"]
            * (self.target_register_length - 1)
            * self.target_register_bit_size
        )
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)
