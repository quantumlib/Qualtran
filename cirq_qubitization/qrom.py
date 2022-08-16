from typing import Tuple, Sequence, Optional
from functools import cached_property
import cirq
from cirq_qubitization import unary_iteration
from cirq_qubitization.gate_with_registers import Registers


class QROM(unary_iteration.UnaryIterationGate):
    """Gate to load data[l] in the target register when the selection register stores integer l."""

    def __init__(self, *data: Sequence[int], target_bitsizes: Optional[Sequence[int]] = None):
        if len(set(len(d) for d in data)) != 1:
            raise ValueError("All data sequences to load must be of equal length.")
        self._data = tuple(tuple(d) for d in data)
        self._selection_bitsize = (len(data[0]) - 1).bit_length()
        if target_bitsizes is None:
            target_bitsizes = [max(d).bit_length() for d in data]
        else:
            assert len(target_bitsizes) == len(data)
            assert all(t >= max(d).bit_length() for t, d in zip(target_bitsizes, data))
        self._target_bitsizes = target_bitsizes

    @cached_property
    def control_registers(self) -> Registers:
        return Registers([])

    @cached_property
    def selection_registers(self) -> Registers:
        return Registers.build(selection=self._selection_bitsize)

    @cached_property
    def target_registers(self) -> Registers:
        return Registers.build(**{f'target{i}': len for i, len in enumerate(self._target_bitsizes)})

    @cached_property
    def iteration_lengths(self) -> Tuple[int, ...]:
        return (len(self._data[0]),)

    @property
    def data(self) -> Tuple[Tuple[int, ...], ...]:
        return self._data

    def __repr__(self) -> str:
        data_repr = ",".join(repr(d) for d in self.data)
        return f"cirq_qubitization.QROM({data_repr}, target_bitsizes={self._target_bitsizes})"

    def nth_operation(
        self, control: cirq.Qid, selection: int, **target_regs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        for i, d in enumerate(self._data):
            target = target_regs[f'target{i}']
            for q, bit in zip(target, f'{d[selection]:0{len(target)}b}'):
                if int(bit):
                    yield cirq.CNOT(control, q)

    def __eq__(self, other: 'QROM'):
        return self.data == other.data and self._target_bitsizes == other._target_bitsizes
