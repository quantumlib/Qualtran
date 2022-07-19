import abc
from typing import Sequence

import cirq

from cirq_qubitization import and_gate
from cirq_qubitization.gate_with_registers import GateWithRegisters, Registers, Register


class UnaryIterationGate(GateWithRegisters):
    @property
    @abc.abstractmethod
    def control_register(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def selection_register(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def target_register(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def iteration_length(self) -> int:
        pass

    @property
    def registers(self) -> Registers:
        return Registers(
            [
                Register("control", self.control_register),
                Register("selection", self.selection_register),
                Register("ancilla", self.selection_register),
                Register("target", self.target_register),
            ]
        )

    @abc.abstractmethod
    def nth_operation(
        self, n: int, control: cirq.Qid, target: Sequence[cirq.Qid]
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
    ) -> cirq.OP_TREE:
        if l >= min(r, self.iteration_length):
            yield []
        if l == (r - 1):
            yield self.nth_operation(l, control, target)
        else:
            assert sl < len(selection)
            m = (l + r) >> 1
            if m >= self.iteration_length:
                yield from self._unary_iteration_segtree(
                    control, selection, ancilla, target, sl + 1, l, m
                )
            else:
                anc, sq = ancilla[sl], selection[sl]
                yield and_gate.And((1, 0)).on(control, sq, anc)
                yield from self._unary_iteration_segtree(
                    anc, selection, ancilla, target, sl + 1, l, m
                )
                yield cirq.CNOT(control, anc)
                yield from self._unary_iteration_segtree(
                    anc, selection, ancilla, target, sl + 1, m, r
                )
                yield and_gate.And(adjoint=True).on(control, sq, anc)

    def _decompose_single_control(
        self,
        control: cirq.Qid,
        selection: Sequence[cirq.Qid],
        ancilla: Sequence[cirq.Qid],
        target: Sequence[cirq.Qid],
    ) -> cirq.OP_TREE:
        assert len(selection) == len(ancilla)
        assert 2 ** len(selection) >= self.iteration_length
        yield from self._unary_iteration_segtree(
            control, selection, ancilla, target, 0, 0, 2 ** len(selection)
        )

    def _decompose_zero_control(
        self,
        selection: Sequence[cirq.Qid],
        ancilla: Sequence[cirq.Qid],
        target: Sequence[cirq.Qid],
    ):
        assert len(selection) == len(ancilla)
        assert 2 ** len(selection) >= self.iteration_length
        assert len(selection) > 0
        sl, l, r = 0, 0, 2 ** len(selection)
        m = (l + r) >> 1
        yield cirq.X(ancilla[0]).controlled_by(selection[0], control_values=[0])
        yield from self._unary_iteration_segtree(
            ancilla[0], selection, ancilla, target, sl + 1, l, m
        )
        yield cirq.X(ancilla[0])
        yield from self._unary_iteration_segtree(
            ancilla[0], selection, ancilla, target, sl + 1, m, r
        )
        yield cirq.CNOT(selection[0], ancilla[0])

    def decompose_from_registers(
        self,
        control: Sequence[cirq.Qid],
        selection: Sequence[cirq.Qid],
        ancilla: Sequence[cirq.Qid],
        target: Sequence[cirq.Qid],
    ) -> cirq.OP_TREE:
        if len(control) == 0:
            yield from self._decompose_zero_control(selection, ancilla, target)
        if len(control) == 1:
            yield from self._decompose_single_control(
                control[0], selection, ancilla, target
            )
        return NotImplemented
