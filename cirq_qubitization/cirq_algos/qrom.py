from functools import cached_property
from typing import Callable, Optional, Sequence, Tuple

import cirq

from cirq_qubitization import cirq_infra
from cirq_qubitization.cirq_algos import and_gate, unary_iteration


@cirq.value_equality
class QROM(unary_iteration.UnaryIterationGate):
    """Gate to load data[l] in the target register when the selection register stores integer l."""

    def __init__(
        self,
        *data: Sequence[int],
        target_bitsizes: Optional[Sequence[int]] = None,
        num_controls: int = 0,
    ):
        if len(set(len(d) for d in data)) != 1:
            raise ValueError("All data sequences to load must be of equal length.")
        self._data = tuple(tuple(d) for d in data)
        self._selection_bitsize = (len(data[0]) - 1).bit_length()
        self._num_controls = num_controls
        if target_bitsizes is None:
            target_bitsizes = [max(d).bit_length() for d in data]
        assert len(target_bitsizes) == len(data)
        assert all(t >= max(d).bit_length() for t, d in zip(target_bitsizes, data))
        self._target_bitsizes = tuple(target_bitsizes)

    @cached_property
    def control_registers(self) -> cirq_infra.Registers:
        return (
            cirq_infra.Registers.build(control=self._num_controls)
            if self._num_controls
            else cirq_infra.Registers([])
        )

    @cached_property
    def selection_registers(self) -> cirq_infra.SelectionRegisters:
        return cirq_infra.SelectionRegisters.build(
            selection=(self._selection_bitsize, len(self._data[0]))
        )

    @cached_property
    def target_registers(self) -> cirq_infra.Registers:
        return cirq_infra.Registers.build(
            **{f'target{i}': len for i, len in enumerate(self._target_bitsizes)}
        )

    @property
    def data(self) -> Tuple[Tuple[int, ...], ...]:
        return self._data

    def __repr__(self) -> str:
        data_repr = ",".join(repr(d) for d in self.data)
        return f"cirq_qubitization.QROM({data_repr}, target_bitsizes={self._target_bitsizes})"

    def _load_nth_data(
        self, n: int, gate: Callable[[cirq.Qid], cirq.Operation], **target_regs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        for i, d in enumerate(self._data):
            target = target_regs[f'target{i}']
            for q, bit in zip(target, f'{d[n]:0{len(target)}b}'):
                if int(bit):
                    yield gate(q)

    def decompose_zero_selection(self, **qubit_regs: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        controls = self.control_registers.merge_qubits(**qubit_regs)
        target_regs = {k: v for k, v in qubit_regs.items() if k in self.target_registers}
        if self._num_controls == 0:
            yield from self._load_nth_data(0, cirq.X, **target_regs)
        elif self._num_controls == 1:
            yield from self._load_nth_data(0, lambda q: cirq.CNOT(controls[0], q), **target_regs)
        else:
            and_ancilla = cirq_infra.qalloc(len(controls) - 2)
            and_target = cirq_infra.qalloc(1)[0]
            multi_controlled_and = and_gate.And((1,) * len(controls)).on_registers(
                control=controls, ancilla=and_ancilla, target=and_target
            )
            yield multi_controlled_and
            yield from self._load_nth_data(0, lambda q: cirq.CNOT(and_target, q), **target_regs)
            yield multi_controlled_and**-1
            cirq_infra.qfree(and_ancilla + [and_target])

    def nth_operation(
        self, control: cirq.Qid, selection: int, **target_regs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        yield from self._load_nth_data(selection, lambda q: cirq.CNOT(control, q), **target_regs)

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["@"] * self._num_controls
        wire_symbols += ["In"] * self._selection_bitsize
        for i, target in enumerate(self.target_registers):
            wire_symbols += [f"QROM_{i}"] * target.bitsize
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def __eq__(self, other: 'QROM'):
        return self.data == other.data and self._target_bitsizes == other._target_bitsizes

    def __pow__(self, power: int):
        if power in [1, -1]:
            return self
        return NotImplemented

    def _value_equality_values_(self):
        return self._selection_bitsize, self._target_bitsizes, self._num_controls, self.data
