from functools import cached_property
from typing import Sequence, Tuple

import cirq
from attrs import frozen

from cirq_qubitization import cirq_infra
from cirq_qubitization.cirq_algos import unary_iteration
from cirq_qubitization.cirq_infra.gate_with_registers import Registers, SelectionRegisters


@frozen
class SelectedMajoranaFermionGate(unary_iteration.UnaryIterationGate):
    """Implements U s.t. U|l>|Psi> -> |l> T_{l} . Z_{l - 1} ... Z_{0} |Psi>

    where T = single qubit target gate. Defaults to pauli Y.


    Args:
        selection_regs: Indexing `select` registers of type `SelectionRegisters`. It also contains
            information about the iteration length of each selection register.
        control_regs: Control registers for constructing a controlled version of the gate.
        target_gate: Single qubit gate to be applied to the target qubits.

    References:
        See Fig 9 of https://arxiv.org/abs/1805.03662 for more details.
    """

    selection_regs: SelectionRegisters
    control_regs: Registers = Registers.build(control=1)
    target_gate: cirq.Gate = cirq.Y

    @classmethod
    def make_on(cls, *, target_gate=cirq.Y, **quregs: Sequence[cirq.Qid]) -> cirq.Operation:
        """Helper constructor to automatically deduce selection_regs attribute."""
        return cls(
            selection_regs=SelectionRegisters.build(
                selection=(len(quregs['selection']), len(quregs['target']))
            ),
            target_gate=target_gate,
        ).on_registers(**quregs)

    @cached_property
    def control_registers(self) -> Registers:
        return self.control_regs

    @cached_property
    def selection_registers(self) -> SelectionRegisters:
        return self.selection_regs

    @cached_property
    def target_registers(self) -> Registers:
        return Registers.build(target=self.selection_regs.total_iteration_size)

    @cached_property
    def iteration_lengths(self) -> Tuple[int, ...]:
        return self.selection_registers.iteration_lengths

    @cached_property
    def extra_registers(self) -> Registers:
        return Registers.build(accumulator=1)

    def decompose_from_registers(self, **qubit_regs: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        qubit_regs['accumulator'] = cirq_infra.qalloc(1)
        yield cirq.X(*qubit_regs['accumulator']).controlled_by(
            *qubit_regs[self.control_regs[0].name]
        )
        yield super().decompose_from_registers(**qubit_regs)
        cirq_infra.qfree(qubit_regs['accumulator'])

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["@"] * self.control_registers.bitsize
        wire_symbols += ["In"] * self.selection_registers.bitsize
        wire_symbols += [f"Z{self.target_gate}"] * self.target_registers.bitsize
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def nth_operation(
        self,
        control: cirq.Qid,
        target: Sequence[cirq.Qid],
        accumulator: Sequence[cirq.Qid],
        **selection_indices: int,
    ) -> cirq.OP_TREE:
        selection_idx = tuple(selection_indices[reg.name] for reg in self.selection_regs)
        target_idx = self.selection_registers.to_flat_idx(*selection_idx)
        yield cirq.CNOT(control, *accumulator)
        yield self.target_gate(target[target_idx]).controlled_by(control)
        yield cirq.CZ(*accumulator, target[target_idx])
