import abc
from functools import cached_property
from typing import Sequence, Tuple, Callable

import cirq

from cirq_qubitization import and_gate
from cirq_qubitization.gate_with_registers import GateWithRegisters, Registers, Register


class SegTreeGate(GateWithRegisters):
    def __init__(
        self,
        *,
        depth: int,
        left: int,
        right: int,
        iteration_length: int,
        nth_operation: Callable,
        selection_bitsize: int,
        ancilla_bitsize: int,
        target_bitsize: int,
    ):
        self.depth = depth
        self.left = left
        self.right = right

        self.iteration_length = iteration_length
        self.nth_operation = nth_operation
        self._selection_bitsize = selection_bitsize
        self._ancilla_bitsize = ancilla_bitsize
        self._target_bitsize = target_bitsize

    def __str__(self):
        return f"SegTree({self.depth}, {self.left}, {self.right})"

    @cached_property
    def registers(self) -> Registers:
        return Registers(
            [
                Register("control", 1),
                Register("selection", self._selection_bitsize),
                Register("ancilla", self._ancilla_bitsize),
                Register("target", self._target_bitsize),
            ]
        )

    def _recurse(self, depth: int, left: int, right: int, pop_ancilla: bool = False):
        if pop_ancilla:
            ancilla_bitsize = self._ancilla_bitsize - 1
        else:
            ancilla_bitsize = self._ancilla_bitsize
        return SegTreeGate(
            depth=depth,
            left=left,
            right=right,
            iteration_length=self.iteration_length,
            nth_operation=self.nth_operation,
            selection_bitsize=self._selection_bitsize,
            ancilla_bitsize=ancilla_bitsize,
            target_bitsize=self._target_bitsize,
        )

    def decompose_from_registers(
        self,
        control: Sequence[cirq.Qid],
        selection: Sequence[cirq.Qid],
        ancilla: Sequence[cirq.Qid],
        target: Sequence[cirq.Qid],
    ) -> cirq.OP_TREE:
        (control,) = control
        if self.left >= min(self.right, self.iteration_length):
            yield []
            return

        if self.left == (self.right - 1):
            yield self.nth_operation(self.left, control, target)
            return

        assert self.depth < len(selection)
        mid = (self.left + self.right) >> 1
        if mid >= self.iteration_length:
            yield self._recurse(depth=self.depth + 1, left=self.left, right=mid).on(
                control, *selection, *ancilla, *target
            )
            return

        anc = ancilla[0]
        new_anc_reg = ancilla[1:]
        sq = selection[self.depth]
        yield and_gate.And((1, 0)).on(control, sq, anc)
        yield self._recurse(depth=self.depth + 1, left=self.left, right=mid, pop_ancilla=True).on(
            anc, *selection, *new_anc_reg, *target
        )
        yield cirq.CNOT(control, anc)
        yield self._recurse(depth=self.depth + 1, left=mid, right=self.right, pop_ancilla=True).on(
            anc, *selection, *new_anc_reg, *target
        )
        yield and_gate.And(adjoint=True).on(control, sq, anc)


class UnaryIterationGate(GateWithRegisters):
    @cached_property
    @abc.abstractmethod
    def control_registers(self) -> Registers:
        pass

    @cached_property
    @abc.abstractmethod
    def selection_registers(self) -> Registers:
        pass

    @cached_property
    @abc.abstractmethod
    def target_registers(self) -> Registers:
        pass

    @cached_property
    @abc.abstractmethod
    def iteration_lengths(self) -> Tuple[int, ...]:
        pass

    @cached_property
    def iteration_length(self) -> int:
        max_iteration_bin = "".join(
            f"{l - 1    :0{r.bitsize}b}"
            for r, l in zip(self.selection_registers, self.iteration_lengths)
        )
        return 1 + int(max_iteration_bin, 2)

    @cached_property
    def ancilla_registers(self) -> Registers:
        control_ancilla_bitsize = max(0, self.control_registers.bitsize - 1)
        iteration_ancilla_bitsize = self.selection_registers.bitsize
        return Registers.build(ancilla=control_ancilla_bitsize + iteration_ancilla_bitsize)

    @cached_property
    def registers(self) -> Registers:
        return Registers(
            [
                *self.control_registers,
                *self.selection_registers,
                *self.ancilla_registers,
                *self.target_registers,
                *self.extra_registers,
            ]
        )

    @cached_property
    def extra_registers(self) -> Registers:
        return Registers([])

    @abc.abstractmethod
    def nth_operation(self, **kwargs) -> cirq.OP_TREE:
        """Apply nth operation on the target registers when selection registers store `n`.

        The `UnaryIterationGate` class is a mixin that represents a coherent for-loop over
        different indices (i.e. selection registers). This method denotes the "body" of the
        for-loop, which is executed `np.prod(self.iteration_lengths)` times and each iteration
        represents a unique combination of values stored in selection registers. For each call,
        the method should return the operations that should be applied to the target registers,
        given the values stored in selection registers.

        The derived classes should specify the following arguments as `**kwargs`:
            1) `control: cirq.Qid`: A qubit which can be used as a control to selectively
            apply operations when selection register stores specific value.
            2) Register names in `self.selection_registers`: Each argument corresponds to
            a selection register and represents the integer value stored in the register.
            3) Register names in `self.target_registers`: Each argument corresponds to a target
            register and represents the sequence of qubits that represent the target register.
            4) Register names in `self.extra_regs`: Each argument corresponds to an extra
            register and represents the sequence of qubits that represent the extra register.
        """

    def _apply_nth_operation(
        self, n: int, control: cirq.Qid, target: Sequence[cirq.Qid], **extra_regs
    ) -> cirq.OP_TREE:
        indices = self.selection_registers.split_integer(n)
        targets = self.target_registers.split_qubits(target)
        all_indices_valid = all(
            indices[r.name] < iter_len
            for r, iter_len in zip(self.selection_registers, self.iteration_lengths)
        )
        yield self.nth_operation(
            control=control, **targets, **indices, **extra_regs
        ) if all_indices_valid else []

    def _get_segtree_gate(self, depth=0, left=0, right=None):
        if right is None:
            right = 2**self.selection_registers.bitsize
        return SegTreeGate(
            depth=depth,
            left=left,
            right=right,
            iteration_length=self.iteration_length,
            nth_operation=self.nth_operation,
            selection_bitsize=self.selection_registers.bitsize,
            ancilla_bitsize=self.ancilla_registers.bitsize,
            target_bitsize=self.target_registers.bitsize,
        )

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
        yield from self._get_segtree_gate().decompose_from_registers(
            control=[control], selection=selection, ancilla=ancilla, target=target
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
        yield from self._get_segtree_gate(depth=sl + 1, left=l, right=m).decompose_from_registers(
            control=[ancilla[0]], selection=selection, ancilla=ancilla, target=target
        )
        yield cirq.X(ancilla[0])
        yield from self._get_segtree_gate(depth=sl + 1, left=m, right=r).decompose_from_registers(
            control=[ancilla[0]], selection=selection, ancilla=ancilla, target=target
        )
        yield cirq.CNOT(selection[0], ancilla[0])

    def decompose_from_registers(self, **qubit_regs: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        control = self.control_registers.merge_qubits(**qubit_regs)
        selection = self.selection_registers.merge_qubits(**qubit_regs)
        target = self.target_registers.merge_qubits(**qubit_regs)
        ancilla = self.ancilla_registers.merge_qubits(**qubit_regs)
        extra_regs = {k: v for k, v in qubit_regs.items() if k in self.extra_registers}

        if len(control) == 0:
            yield from self._decompose_zero_control(selection, ancilla, target, **extra_regs)
        elif len(control) == 1:
            yield from self.decompose_single_control(
                control[0], selection, ancilla, target, **extra_regs
            )
        else:
            control_bitsize = self.control_registers.bitsize
            and_ancillas = ancilla[: control_bitsize - 2]
            and_target = ancilla[control_bitsize - 2]
            selection_ancillas = ancilla[control_bitsize - 1 :]
            multi_controlled_and = and_gate.And((1,) * len(control)).on_registers(
                control=control, ancilla=and_ancillas, target=and_target
            )
            yield multi_controlled_and
            yield from self.decompose_single_control(
                and_target, selection, selection_ancillas, target, **extra_regs
            )
            yield multi_controlled_and**-1
