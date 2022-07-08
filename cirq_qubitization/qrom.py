import cirq
from typing import Optional, Sequence
from cirq_qubitization import unary_iteration


class QROM(unary_iteration.UnaryIterationGate):
    """Gate to load data[l] in the target_register when selection_register stores integer l."""

    def __init__(
        self,
        data: Sequence[int],
        *,
        selection_register: Optional[int] = None,
        target_register: Optional[int] = None,
    ):
        if selection_register is None:
            selection_register = len(data).bit_length()
        if target_register is None:
            target_register = max(data).bit_length()
        assert 2**selection_register >= len(data)
        assert 2**target_register >= max(data)
        self._data = data
        self._selection_register = selection_register
        self._target_register = target_register

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
        return len(self._data)

    def nth_operation(
        self, n: int, control: cirq.Qid, target: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        for i, bit in enumerate(format(self._data[n], f"0{len(target)}b")):
            if bit == "1":
                yield cirq.CNOT(control, target[i])
