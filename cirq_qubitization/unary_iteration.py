import abc
from typing import Sequence, List

import cirq

from cirq_qubitization import and_gate


class UnaryIterationGate(cirq.Gate):
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

    @abc.abstractmethod
    def nth_operation(
        self, n: int, control: cirq.Qid, target: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        pass

    def _num_qubits_(self) -> int:
        return (
            self.control_register + 2 * self.selection_register + self.target_register
        )

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

    def decompose_single_control(
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

    def _decompose_(self, qubits: List[cirq.Qid]) -> cirq.OP_TREE:
        control = qubits[: self.control_register]
        selection = qubits[
            self.control_register : self.control_register + self.selection_register
        ]
        ancilla = qubits[
            self.control_register
            + self.selection_register : self.control_register
            + 2 * self.selection_register
        ]
        target = qubits[self.control_register + 2 * self.selection_register :]
        if len(control) == 0:
            yield from self._decompose_zero_control(selection, ancilla, target)
        if len(control) == 1:
            yield from self.decompose_single_control(
                control[0], selection, ancilla, target
            )
        return NotImplemented
