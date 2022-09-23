from functools import cached_property
from typing import Any, Sequence
import cirq
from cirq_qubitization import multi_target_cnot
from cirq_qubitization.gate_with_registers import GateWithRegisters, Registers


class MultiTargetCSwap(GateWithRegisters):
    """Implements multi-target controlled swap unitary $CSWAP_{n} = |0><0| I + |1><1| SWAP_{n}$."""

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
    """Approximately implements a multi-target controlled swap unitary using only 4 * N T-gates.

    Implements the unitary $CSWAP_{n} = |0><0| I + |1><1| SWAP_{n}$ such that the output state is
    correct up to a global phase factor of +1 / -1.

    This is useful when the incorrect phase can be absorbed in a garbage state of an algorithm; and
    thus ignored. See Appendix B.2.c of https://arxiv.org/abs/1812.00954 for more details.
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


class SwapWithZeroGate(GateWithRegisters):
    """Swaps |Psi_0> with |Psi_x> if selection register stores index `x`.

    Implements the unitary U |x> |Psi_0> |Psi_1> ... |Psi_{n-1}> --> |x> |Psi_x> |Rest of Psi>.
    Note that the state of `|Rest of Psi>` is allowed to be anything and should not be depended
    upon.

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis]
        (https://arxiv.org/abs/1812.00954).
            Low, Kliuchnikov, Schaeffer. 2018.
    """

    def __init__(self, selection_bitsize: int, target_bitsize: int, n_target_registers: int):
        assert n_target_registers <= 2**selection_bitsize
        self._selection_bitsize = selection_bitsize
        self._target_bitsize = target_bitsize
        self._n_target_registers = n_target_registers

    @cached_property
    def selection_registers(self) -> Registers:
        return Registers.build(selection=self._selection_bitsize)

    @cached_property
    def target_registers(self) -> Registers:
        return Registers.build(
            **{f'target{i}': self._target_bitsize for i in range(self._n_target_registers)}
        )

    @cached_property
    def registers(self) -> Registers:
        return Registers([*self.selection_registers, *self.target_registers])

    def decompose_from_registers(
        self, selection: Sequence[cirq.Qid], **target_regs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        assert len(target_regs) == self._n_target_registers
        cswap_n = MultiTargetCSwapApprox(self._target_bitsize)
        for j in range(len(selection)):
            for i in range(self._n_target_registers - 2**j):
                yield cswap_n.on_registers(
                    control=selection[len(selection) - j - 1],
                    target_x=target_regs[f'target{i}'],
                    target_y=target_regs[f'target{i + 2**j}'],
                )

    def __repr__(self) -> str:
        return (
            "cirq_qubitization.SwapWithZeroGate("
            f"{self.selection_bitsize},"
            f"{self.target_bitsize},"
            f"{self.n_target_registers}"
            f")"
        )

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["@(r⇋0)"] * self._selection_bitsize
        for i in range(self._n_target_registers):
            wire_symbols += [f"swap_{i}"] * self._target_bitsize
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)
