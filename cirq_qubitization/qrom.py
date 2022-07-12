import cirq
from typing import Union, Sequence
from cirq_qubitization import unary_iteration


class QROM(unary_iteration.UnaryIterationGate):
    """Gate to load data[l] in the target_register when selection_register stores integer l."""

    def __init__(self, *data: Sequence[int]):
        if len(set(len(d) for d in data)) != 1:
            raise ValueError("All data sequences to load must be of equal length.")
        self._data = data
        self._selection_register = len(data[0]).bit_length()
        self._individual_target_registers = [max(d).bit_length() for d in data]
        self._target_register = sum(self._individual_target_registers)

    @property
    def control_register(self) -> int:
        return 1

    @property
    def selection_register(self) -> int:
        return self._selection_register

    @property
    def target_register(self) -> int:
        return self._target_register

    @property
    def iteration_length(self) -> int:
        return len(self._data[0])

    def on(
        self,
        *,
        control_register: Union[cirq.Qid, Sequence[cirq.Qid]],
        selection_register: Sequence[cirq.Qid],
        selection_ancilla: Sequence[cirq.Qid],
        target_register: Union[Sequence[cirq.Qid], Sequence[Sequence[cirq.Qid]]],
    ) -> cirq.Operation:
        if isinstance(control_register, cirq.Qid):
            control_register = [control_register]
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
            list(control_register)
            + list(selection_register)
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
