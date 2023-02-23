from itertools import count

import cirq


class CleanQubit(cirq.Qid):
    _ids = count(0)

    def __init__(self):
        self.id = next(self._ids)

    def _comparison_key(self) -> int:
        return self.id

    @property
    def dimension(self) -> int:
        return 2

    @classmethod
    def reset_count(cls) -> None:
        cls._ids = count(0)

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        return cirq.CircuitDiagramInfo(wire_symbols=(f"_c{self.id}",))


class BorrowableQubit(cirq.Qid):
    _ids = count(0)

    def __init__(self):
        self.id = next(self._ids)

    def _comparison_key(self) -> int:
        return self.id

    @property
    def dimension(self) -> int:
        return 2

    @classmethod
    def reset_count(cls) -> None:
        cls._ids = count(0)

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        return cirq.CircuitDiagramInfo(wire_symbols=(f"_b{self.id}",))
