import abc
from functools import cached_property
from typing import Sequence

import cirq

from cirq_qubitization import and_gate
from cirq_qubitization.gate_with_registers import GateWithRegisters, Registers, Register


class UnaryIterationGate(GateWithRegisters):
    @property
    @abc.abstractmethod
    def control_bitsize(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def selection_bitsize(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def target_bitsize(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def iteration_length(self) -> int:
        pass

    @property
    def ancilla_bitsize(self) -> int:
        return max(0, self.control_bitsize - 1) + self.selection_bitsize

    @cached_property
    def registers(self) -> Registers:
        return Registers(
            [
                Register("control", self.control_bitsize),
                Register("selection", self.selection_bitsize),
                Register("ancilla", self.ancilla_bitsize),
                Register("target", self.target_bitsize),
                *self.extra_registers,
            ]
        )

    @cached_property
    def extra_registers(self) -> Sequence[Register]:
        return []

    @abc.abstractmethod
    def nth_operation(
        self,
        n: int,
        control: cirq.Qid,
        target: Sequence[cirq.Qid],
        **extra_regs: Sequence[cirq.Qid],
    ) -> cirq.OP_TREE:
        pass

    def _unary_iteration_segtree(
        self,
        control: cirq.Qid,
        selection: Sequence[cirq.Qid],
        ancilla: Sequence[cirq.Qid],
        target: Sequence[cirq.Qid],
        sl: int,
        l: int,
        r: int,
        **extra_regs: Sequence[cirq.Qid],
    ) -> cirq.OP_TREE:
        if l >= min(r, self.iteration_length):
            yield []
        if l == (r - 1):
            yield self.nth_operation(l, control, target, **extra_regs)
        else:
            assert sl < len(selection)
            m = (l + r) >> 1
            if m >= self.iteration_length:
                yield from self._unary_iteration_segtree(
                    control, selection, ancilla, target, sl + 1, l, m, **extra_regs
                )
            else:
                anc, sq = ancilla[sl], selection[sl]
                yield and_gate.And((1, 0)).on(control, sq, anc)
                yield from self._unary_iteration_segtree(
                    anc, selection, ancilla, target, sl + 1, l, m, **extra_regs
                )
                yield cirq.CNOT(control, anc)
                yield from self._unary_iteration_segtree(
                    anc, selection, ancilla, target, sl + 1, m, r, **extra_regs
                )
                yield and_gate.And(adjoint=True).on(control, sq, anc)

    def decompose_single_control(
        self,
        control: cirq.Qid,
        selection: Sequence[cirq.Qid],
        ancilla: Sequence[cirq.Qid],
        target: Sequence[cirq.Qid],
        **extra_regs: Sequence[cirq.Qid],
    ) -> cirq.OP_TREE:
        assert len(selection) == len(ancilla)
        assert 2 ** len(selection) >= self.iteration_length
        yield from self._unary_iteration_segtree(
            control, selection, ancilla, target, 0, 0, 2 ** len(selection), **extra_regs
        )

    def _decompose_zero_control(
        self,
        selection: Sequence[cirq.Qid],
        ancilla: Sequence[cirq.Qid],
        target: Sequence[cirq.Qid],
        **extra_regs: Sequence[cirq.Qid],
    ):
        assert len(selection) == len(ancilla)
        assert 2 ** len(selection) >= self.iteration_length
        assert len(selection) > 0
        sl, l, r = 0, 0, 2 ** len(selection)
        m = (l + r) >> 1
        yield cirq.X(ancilla[0]).controlled_by(selection[0], control_values=[0])
        yield from self._unary_iteration_segtree(
            ancilla[0], selection, ancilla, target, sl + 1, l, m, **extra_regs
        )
        yield cirq.X(ancilla[0])
        yield from self._unary_iteration_segtree(
            ancilla[0], selection, ancilla, target, sl + 1, m, r, **extra_regs
        )
        yield cirq.CNOT(selection[0], ancilla[0])

    def decompose_from_registers(
        self,
        control: Sequence[cirq.Qid],
        selection: Sequence[cirq.Qid],
        ancilla: Sequence[cirq.Qid],
        target: Sequence[cirq.Qid],
        **extra_regs: Sequence[cirq.Qid],
    ) -> cirq.OP_TREE:
        if len(control) == 0:
            yield from self._decompose_zero_control(selection, ancilla, target, **extra_regs)
        elif len(control) == 1:
            yield from self.decompose_single_control(
                control[0], selection, ancilla, target, **extra_regs
            )
        else:
            and_ancillas = ancilla[: self.control_bitsize - 2]
            and_target = ancilla[self.control_bitsize - 2]
            selection_ancillas = ancilla[self.control_bitsize - 1 :]
            multi_controlled_and = and_gate.And((1,) * len(control)).on_registers(
                control=control, ancilla=and_ancillas, target=and_target
            )
            yield multi_controlled_and
            yield from self.decompose_single_control(
                and_target, selection, selection_ancillas, target, **extra_regs
            )
            yield multi_controlled_and**-1
