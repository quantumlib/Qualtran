import cirq
from typing import Tuple, Union, Sequence, Optional
from cirq_qubitization import unary_iteration


class QROM(unary_iteration.UnaryIterationGate):
    """Gate to load data[l] in the target_register when selection_register stores integer l."""

    def __init__(
        self, *data: Sequence[int], target_registers: Optional[Sequence[int]] = None
    ):
        if len(set(len(d) for d in data)) != 1:
            raise ValueError("All data sequences to load must be of equal length.")
        self._data = tuple(tuple(d) for d in data)
        self._selection_register = (len(data[0]) - 1).bit_length()
        if target_registers is None:
            target_registers = [max(d).bit_length() for d in data]
        else:
            assert len(target_registers) == len(data)
            assert all(t >= max(d).bit_length() for t, d in zip(target_registers, data))
        self._individual_target_registers = target_registers
        self._target_register = sum(self._individual_target_registers)

    @property
    def control_register(self) -> int:
        return 0

    @property
    def selection_register(self) -> int:
        return self._selection_register

    @property
    def target_register(self) -> int:
        return self._target_register

    @property
    def iteration_length(self) -> int:
        return len(self._data[0])

    @property
    def data(self) -> Tuple[Tuple[int, ...], ...]:
        return self._data

    def __repr__(self) -> str:
        return f"cirq_qubitization.QROM({self.data})"

    def on_registers(
        self,
        *,
        selection_register: Sequence[cirq.Qid],
        selection_ancilla: Sequence[cirq.Qid],
        target_register: Union[Sequence[cirq.Qid], Sequence[Sequence[cirq.Qid]]],
    ) -> cirq.GateOperation:
        if not isinstance(target_register[0], cirq.Qid):
            assert (
                len(t) == tr
                for t, tr in zip(target_register, self._individual_target_registers)
            ), f"Length of each target register must match {self._individual_target_registers}"
            flat_target_register = [t for target in target_register for t in target]
        else:
            flat_target_register = target_register
        assert len(flat_target_register) == self.target_register
        return cirq.GateOperation(
            self,
            list(selection_register)
            + list(selection_ancilla)
            + list(flat_target_register),
        )

    def nth_operation(
        self, n: int, control: cirq.Qid, target: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        offset = 0
        for d, target_length in zip(self._data, self._individual_target_registers):
            for i, bit in enumerate(format(d[n], f"0{target_length}b")):
                if bit == "1":
                    yield cirq.CNOT(control, target[offset + i])
            offset += target_length
