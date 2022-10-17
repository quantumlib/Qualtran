from typing import Sequence, Tuple
from functools import cached_property
import cirq
from cirq_qubitization.gate_with_registers import Registers
from cirq_qubitization import unary_iteration


class SelectedMajoranaFermionGate(unary_iteration.UnaryIterationGate):
    """Implements U s.t. U|l>|Psi> -> |l> T_{l} . Z_{l - 1} ... Z_{0} |Psi>

    where T = single qubit target gate. Defaults to pauli Y.

    Uses:
    * 1 Control qubit.
    * 1 Accumulator qubit.
    * `selection_bitsize` number of selection qubits.
    * `target_bitsize` number of target qubits.

    See Fig 9 of https://arxiv.org/abs/1805.03662 for more details.
    """

    def __init__(self, selection_bitsize: int, target_bitsize: int, target_gate=cirq.Y):
        self._selection_bitsize = selection_bitsize
        self._target_bitsize = target_bitsize
        self._target_gate = target_gate

    @cached_property
    def control_registers(self) -> Registers:
        return Registers.build(control=1)

    @cached_property
    def selection_registers(self) -> Registers:
        return Registers.build(selection=self._selection_bitsize)

    @cached_property
    def target_registers(self) -> Registers:
        return Registers.build(target=self._target_bitsize)

    @cached_property
    def iteration_lengths(self) -> Tuple[int, ...]:
        return (self._target_bitsize,)

    @cached_property
    def extra_registers(self) -> Registers:
        return Registers.build(accumulator=1)

    def _decompose_single_control(
        self,
        control: cirq.Qid,
        selection: Sequence[cirq.Qid],
        ancilla: Sequence[cirq.Qid],
        target: Sequence[cirq.Qid],
        accumulator: Sequence[cirq.Qid],
    ) -> cirq.OP_TREE:
        yield cirq.CNOT(control, accumulator[0])
        yield from super()._decompose_single_control(
            control, selection, ancilla, target, accumulator=accumulator
        )

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["@"] * self.control_registers.total_size
        wire_symbols += ["In"] * self.selection_registers.total_size
        wire_symbols += ["Anc"] * self.ancilla_registers.total_size
        wire_symbols += [f"Z{self._target_gate}"] * self.target_registers.total_size
        wire_symbols += ["Acc"]
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def nth_operation(
        self,
        selection: int,
        control: cirq.Qid,
        target: Sequence[cirq.Qid],
        accumulator: Sequence[cirq.Qid],
    ) -> cirq.OP_TREE:
        yield cirq.CNOT(control, accumulator[0])
        yield self._target_gate(target[selection]).controlled_by(control)
        yield cirq.Z(target[selection]).controlled_by(accumulator[0])
