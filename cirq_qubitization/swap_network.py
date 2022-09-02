from functools import cached_property
from typing import Any, Sequence
import cirq
from cirq_qubitization import multi_target_cnot
from cirq_qubitization.gate_with_registers import GateWithRegisters, Registers


class MultiTargetCSwap(GateWithRegisters):
    """Implements a multi-target controlled swap unitary $CSWAP_n = |0><0| I + |1><1| SWAP_n$.

    This decomposes into a qubitwise SWAP on the two target registers, and takes 14*n T-gates.

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis](https://arxiv.org/abs/1812.00954).
        Low et. al. 2018. See Appendix B.2.c.
    """

    def __init__(self, target_bitsize: int) -> None:
        self._target_bitsize = target_bitsize

    @cached_property
    def registers(self) -> Registers:
        return Registers.build(
            control=1, target_x=self._target_bitsize, target_y=self._target_bitsize
        )

    def decompose_from_registers(
        self,
        control: Sequence[cirq.Qid],
        target_x: Sequence[cirq.Qid],
        target_y: Sequence[cirq.Qid],
    ) -> cirq.OP_TREE:
        (control,) = control
        yield [cirq.CSWAP(control, t_x, t_y) for t_x, t_y in zip(target_x, target_y)]

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        if not args.use_unicode_characters:
            return cirq.CircuitDiagramInfo(
                ("@",) + ("swap_x",) * self._target_bitsize + ("swap_y",) * self._target_bitsize
            )
        return cirq.CircuitDiagramInfo(
            ("@",) + ("×(x)",) * self._target_bitsize + ("×(y)",) * self._target_bitsize
        )

    def __repr__(self) -> str:
        return f"cirq_qubitization.MultiTargetCSwap({self._target_bitsize})"

    def __eq__(self, other: 'MultiTargetCSwap') -> Any:
        return type(self) == type(other) and self._target_bitsize == other._target_bitsize


class MultiTargetCSwapApprox(MultiTargetCSwap):
    """Approximately implements a multi-target controlled swap unitary using only 4 * n T-gates.

    Implements the unitary $CSWAP_n = |0><0| I + |1><1| SWAP_n$ such that the output state is
    correct up to a global phase factor of +1 / -1.

    This is useful when the incorrect phase can be absorbed in a garbage state of an algorithm; and
    thus ignored, see the reference for more details.

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis](https://arxiv.org/abs/1812.00954).
        Low et. al. 2018. See Appendix B.2.c.
    """

    def decompose_from_registers(
        self,
        control: Sequence[cirq.Qid],
        target_x: Sequence[cirq.Qid],
        target_y: Sequence[cirq.Qid],
    ) -> cirq.OP_TREE:
        (control,) = control

        def g(q: cirq.Qid, adjoint=False) -> cirq.OP_TREE:
            yield [cirq.S(q), cirq.H(q)]
            yield cirq.T(q) ** (1 - 2 * adjoint)
            yield [cirq.H(q), cirq.S(q) ** -1]

        cnot_x_to_y = [cirq.CNOT(x, y) for x, y in zip(target_x, target_y)]
        cnot_y_to_x = [cirq.CNOT(y, x) for x, y in zip(target_x, target_y)]
        g_inv_on_y = [list(g(q, True)) for q in target_y]  # Uses len(target_y) T-gates
        g_on_y = [list(g(q)) for q in target_y]  # Uses len(target_y) T-gates

        yield [cnot_y_to_x, g_inv_on_y, cnot_x_to_y, g_inv_on_y]
        yield multi_target_cnot.MultiTargetCNOT(len(target_y)).on(control, *target_y)
        yield [g_on_y, cnot_x_to_y, g_on_y, cnot_y_to_x]

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        if not args.use_unicode_characters:
            return cirq.CircuitDiagramInfo(
                ("@(approx)",)
                + ("swap_x",) * self._target_bitsize
                + ("swap_y",) * self._target_bitsize
            )
        return cirq.CircuitDiagramInfo(
            ("@(approx)",) + ("×(x)",) * self._target_bitsize + ("×(y)",) * self._target_bitsize
        )

    def __repr__(self) -> str:
        return f"cirq_qubitization.MultiTargetCSwapApprox({self._target_bitsize})"


class SwapWithZeroGate(cirq.Gate):
    """Swaps |Psi_0> with |Psi_x> if selection register stores index `x`.

    Implements the unitary U |x> |Psi_0> |Psi_1> ... |Psi_n> --> |x> |Psi_x> |Rest of Psi>.
    Note that the state of `|Rest of Psi>` is allowed to be anything and should not be depended
    upon.

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis]
        (https://arxiv.org/abs/1812.00954).
            Low, Kliuchnikov, Schaeffer. 2018.
    """

    def __init__(self, selection_bitsize: int, target_bitsize: int, n_target_registers: int):
        assert n_target_registers <= 2**selection_bitsize
        self.selection_bitsize = selection_bitsize
        self.target_bitsize = target_bitsize
        self.n_target_registers = n_target_registers

    def _num_qubits_(self) -> int:
        return self.selection_bitsize + self.n_target_registers * self.target_bitsize

    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        selection = qubits[: self.selection_bitsize]
        target = [
            qubits[st : st + self.target_bitsize]
            for st in range(self.selection_bitsize, len(qubits), self.target_bitsize)
        ]
        assert len(target) == self.n_target_registers
        swap_n = MultiTargetCSwapApprox(self.target_bitsize)
        for j in range(len(selection)):
            for i in range(len(target) - 2**j):
                yield swap_n.on_registers(
                    control=selection[len(selection) - j - 1],
                    target_x=target[i],
                    target_y=target[i + 2**j],
                )

    def on_registers(
        self, *, selection: Sequence[cirq.Qid], target: Sequence[Sequence[cirq.Qid]]
    ) -> cirq.GateOperation:
        assert len(selection) == self.selection_bitsize
        assert len(target) == self.n_target_registers
        assert all(len(t) == self.target_bitsize for t in target)
        flat_target = [q for t in target for q in t]
        return cirq.GateOperation(self, selection + flat_target)

    def __repr__(self) -> str:
        return (
            "cirq_qubitization.SwapWithZeroGate("
            f"{self.selection_bitsize},"
            f"{self.target_bitsize},"
            f"{self.n_target_registers}"
            f")"
        )

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["@(n)"] * self.selection_bitsize
        wire_symbols += ["swap_0"] * self.target_bitsize
        wire_symbols += ["swap_r"] * (self.n_target_registers - 1) * self.target_bitsize
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)
