from typing import Union, Sequence
import cirq
from cirq_qubitization import unary_iteration


class SelectedMajoranaFermionGate(unary_iteration.UnaryIterationGate):
    """Implements U s.t. U|l>|Psi> -> |l> Y_{l} . Z_{l - 1} ... Z_{0} |Psi>

    Uses:
    * 1 Control qubit.
    * 1 Accumulator qubit.
    * `selection_bitsize` number of selection qubits.
    * `target_bitsize` number of target qubits.

    See Fig 9 of https://arxiv.org/abs/1805.03662 for more details.
    """

    def __init__(self, selection_bitsize: int, target_bitsize: int):
        self._selection_bitsize = selection_bitsize
        self._target_bitsize = target_bitsize

    @property
    def control_bitsize(self) -> int:
        return 1

    @property
    def selection_bitsize(self) -> int:
        return self._selection_bitsize

    @property
    def target_bitsize(self) -> int:
        """First qubit is used as an accumulator and remaining qubits are used as target."""
        return 1 + self._target_bitsize

    @property
    def iteration_length(self) -> int:
        return self._target_bitsize

    def _decompose_single_control(
        self,
        control: cirq.Qid,
        selection: Sequence[cirq.Qid],
        ancilla: Sequence[cirq.Qid],
        target: Sequence[cirq.Qid],
    ) -> cirq.OP_TREE:
        yield cirq.CNOT(control, target[0])
        yield from super()._decompose_single_control(
            control, selection, ancilla, target
        )

    def on_registers(
        self,
        *,
        control_register: Union[cirq.Qid, Sequence[cirq.Qid]],
        selection_register: Sequence[cirq.Qid],
        selection_ancilla: Sequence[cirq.Qid],
        accumulator: cirq.Qid,
        target_register: Sequence[cirq.Qid]
    ) -> cirq.GateOperation:
        if isinstance(control_register, cirq.Qid):
            control_register = [control_register]
        return cirq.GateOperation(
            self,
            list(control_register)
            + list(selection_register)
            + list(selection_ancilla)
            + [accumulator]
            + list(target_register),
        )

    def nth_operation(
        self, n: int, control: cirq.Qid, target: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        accumulator, target_register = target[0], target[1:]
        yield cirq.CNOT(control, accumulator)
        yield cirq.Y(target_register[n]).controlled_by(control)
        yield cirq.Z(target_register[n]).controlled_by(accumulator)
