from typing import Union, Sequence
from functools import cached_property
import cirq
from cirq_qubitization.gate_with_registers import Registers, Register
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

    @property
    def control_bitsize(self) -> int:
        return 1

    @property
    def selection_bitsize(self) -> int:
        return self._selection_bitsize

    @property
    def target_bitsize(self) -> int:
        """First qubit is used as an accumulator and remaining qubits are used as target."""
        return self._target_bitsize

    @property
    def iteration_length(self) -> int:
        return self._target_bitsize

    @cached_property
    def extra_registers(self) -> Sequence[Register]:
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

    def _circuit_diagram_info_(
        self, args: cirq.CircuitDiagramInfoArgs
    ) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["@"] * self.control_bitsize
        wire_symbols += ["In"] * self.selection_bitsize
        wire_symbols += ["Anc"] * self.selection_bitsize
        wire_symbols += [f"Z{self._target_gate}"] * self.target_bitsize
        wire_symbols += ["Acc"]
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def nth_operation(
        self,
        n: int,
        control: cirq.Qid,
        target: Sequence[cirq.Qid],
        accumulator: Sequence[cirq.Qid],
    ) -> cirq.OP_TREE:
        yield cirq.CNOT(control, accumulator[0])
        yield self._target_gate(target[n]).controlled_by(control)
        yield cirq.Z(target[n]).controlled_by(accumulator[0])
