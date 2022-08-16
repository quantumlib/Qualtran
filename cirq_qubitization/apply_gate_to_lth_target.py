from typing import Callable, Sequence, Tuple
from functools import cached_property
import cirq
from cirq_qubitization.unary_iteration import UnaryIterationGate
from cirq_qubitization.gate_with_registers import Registers


class ApplyGateToLthQubit(UnaryIterationGate):
    def __init__(
        self,
        selection_bitsize: int,
        target_bitsize: int,
        nth_gate: Callable[[int], cirq.Gate],
        *,
        control_bitsize: int = 1,
    ):
        self._nth_gate = nth_gate
        self._selection_bitsize = selection_bitsize
        self._target_bitsize = target_bitsize
        self._control_bitsize = control_bitsize

    @cached_property
    def control_registers(self) -> Registers:
        return Registers.build(control=self._control_bitsize)

    @cached_property
    def selection_registers(self) -> Registers:
        return Registers.build(selection=self._selection_bitsize)

    @cached_property
    def target_registers(self) -> Registers:
        return Registers.build(target=self._target_bitsize)

    @cached_property
    def iteration_lengths(self) -> Tuple[int, ...]:
        return (self._target_bitsize,)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["@"] * self.control_registers.bitsize
        wire_symbols += ["In"] * self.selection_registers.bitsize
        wire_symbols += ["Anc"] * self.ancilla_registers.bitsize
        wire_symbols += [str(self._nth_gate(i)) for i in range(self._target_bitsize)]
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def nth_operation(
        self, selection: int, control: cirq.Qid, target: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        return self._nth_gate(selection).on(target[-(selection + 1)]).controlled_by(control)
