from typing import Callable, Sequence, Union
import cirq
from cirq_qubitization.unary_iteration import UnaryIterationGate


class ApplyGateToLthQubit(UnaryIterationGate):
    def __init__(
        self,
        selection_bitsize: int,
        target_bitsize: int,
        nth_gate: Union[cirq.Gate, Callable[[int], cirq.Gate]],
        *,
        control_bitsize: int = 1,
    ):
        self._nth_gate = (
            (lambda _: nth_gate) if isinstance(nth_gate, cirq.Gate) else nth_gate
        )
        self._selection_bitsize = selection_bitsize
        self._target_bitsize = target_bitsize
        self._control_bitsize = control_bitsize

    @property
    def control_bitsize(self) -> int:
        return self._control_bitsize

    @property
    def selection_bitsize(self) -> int:
        return self._selection_bitsize

    @property
    def target_bitsize(self) -> int:
        return self._target_bitsize

    @property
    def iteration_length(self) -> int:
        return self._target_bitsize

    def _circuit_diagram_info_(
        self, args: cirq.CircuitDiagramInfoArgs
    ) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["@"] * self.control_bitsize
        wire_symbols += ["In"] * self.selection_bitsize
        wire_symbols += ["Anc"] * self.selection_bitsize
        wire_symbols += [str(self._nth_gate(i)) for i in range(self.target_bitsize)]
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def nth_operation(
        self, n: int, control: cirq.Qid, target: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        return self._nth_gate(n).on(target[-(n + 1)]).controlled_by(control)
