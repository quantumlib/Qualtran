from functools import cached_property
from typing import Callable, Optional, Sequence, Tuple

import cirq
import numpy as np
from attrs import frozen
from numpy.typing import NDArray

from cirq_qubitization import cirq_infra
from cirq_qubitization.cirq_algos import and_gate, unary_iteration


@frozen
class QROM(unary_iteration.UnaryIterationGate):
    """Gate to load n-dimensional data in the target register when the selection
    register stores the index of the data element to load.
    """

    data: Sequence[NDArray[np.int_]]
    selection_bitsizes: Sequence[int]
    target_bitsizes: Sequence[int]
    num_controls: int = 0

    @classmethod
    def build(cls, data: Sequence[NDArray[np.int_]], num_controls: int = 0) -> 'QROM':
        selection_bitsizes = [(s - 1).bit_length() for s in data[0].shape]
        target_bitsizes = [max(int(np.max(d)).bit_length(), 1) for d in data]
        return QROM(
            data=data,
            selection_bitsizes=selection_bitsizes,
            target_bitsizes=target_bitsizes,
            num_controls=num_controls,
        )

    def __attrs_post_init__(self):
        sizes = [np.prod(s.shape) for s in self.data]
        assert all([isinstance(s, int) for s in self.selection_bitsizes])
        assert all([isinstance(t, int) for t in self.target_bitsizes])
        assert len(set(sizes)) == 1, f"Data must all have the same size: {sizes}"
        assert len(self.target_bitsizes) == len(self.data)
        assert all(
            t >= int(np.max(d)).bit_length() for t, d in zip(self.target_bitsizes, self.data)
        )

    def __hash__(self):
        return hash(tuple(tuple(d.ravel()) for d in self.data))

    @cached_property
    def control_registers(self) -> cirq_infra.Registers:
        return (
            cirq_infra.Registers.build(control=self.num_controls)
            if self.num_controls
            else cirq_infra.Registers([])
        )

    @cached_property
    def selection_registers(self) -> cirq_infra.SelectionRegisters:
        if len(self.data[0].shape) == 1:
            return cirq_infra.SelectionRegisters.build(
                selection=(self.selection_bitsizes[0], self.data[0].shape[0])
            )
        else:
            return cirq_infra.SelectionRegisters.build(
                **{
                    f'selection{i}': (sb, len)
                    for i, (len, sb) in enumerate(zip(self.data[0].shape, self.selection_bitsizes))
                }
            )

    @cached_property
    def target_registers(self) -> cirq_infra.Registers:
        return cirq_infra.Registers.build(
            **{f'target{i}': len for i, len in enumerate(self.target_bitsizes)}
        )

    def __repr__(self) -> str:
        data_repr = "[" + "".join(f"np.{repr(d)}," for d in self.data) + "]"
        selection_repr = repr(self.selection_bitsizes)
        target_repr = repr(self.target_bitsizes)
        return f"cirq_qubitization.QROM({data_repr}, selection_bitsizes={selection_repr}, target_bitsizes={target_repr}, num_controls={self.num_controls})"

    def _load_nth_data(
        self,
        selection_idx: Tuple[int, ...],
        gate: Callable[[cirq.Qid], cirq.Operation],
        **target_regs: Sequence[cirq.Qid],
    ) -> cirq.OP_TREE:
        for i, d in enumerate(self.data):
            target = target_regs[f'target{i}']
            for q, bit in zip(target, f'{int(d[selection_idx]):0{len(target)}b}'):
                if int(bit):
                    yield gate(q)

    def decompose_zero_selection(self, **qubit_regs: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        controls = self.control_registers.merge_qubits(**qubit_regs)
        target_regs = {k: v for k, v in qubit_regs.items() if k in self.target_registers}
        num_indx = len(self.data[0].shape)
        if self.num_controls == 0:
            yield from self._load_nth_data((0,) * num_indx, cirq.X, **target_regs)
        elif self.num_controls == 1:
            yield from self._load_nth_data(
                (0,) * num_indx, lambda q: cirq.CNOT(controls[0], q), **target_regs
            )
        else:
            and_ancilla = cirq_infra.qalloc(len(controls) - 2)
            and_target = cirq_infra.qalloc(1)[0]
            multi_controlled_and = and_gate.And((1,) * len(controls)).on_registers(
                control=controls, ancilla=and_ancilla, target=and_target
            )
            yield multi_controlled_and
            yield from self._load_nth_data(
                (0,) * num_indx, lambda q: cirq.CNOT(and_target, q), **target_regs
            )
            yield multi_controlled_and**-1
            cirq_infra.qfree(and_ancilla + [and_target])

    def nth_operation(self, control: cirq.Qid, **qubit_regs: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        selection_idx = tuple(qubit_regs[reg.name] for reg in self.selection_registers)
        target_regs = {k: v for k, v in qubit_regs.items() if k in self.target_registers}
        yield from self._load_nth_data(
            selection_idx, lambda q: cirq.CNOT(control, q), **target_regs
        )

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["@"] * self.num_controls
        wire_symbols += ["In"] * self.selection_registers.bitsize
        for i, target in enumerate(self.target_registers):
            wire_symbols += [f"QROM_{i}"] * target.bitsize
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def __pow__(self, power: int):
        if power in [1, -1]:
            return self
        return NotImplemented

    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            return NotImplemented
        else:
            return (
                np.array_equal(self.data, other.data)
                and np.array_equal(self.selection_bitsizes, other.selection_bitsizes)
                and np.array_equal(self.target_bitsizes, other.target_bitsizes)
                and np.array_equal(self.num_controls, other.num_controls)
            )
