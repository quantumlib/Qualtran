from typing import Callable, Sequence, Tuple
from functools import cached_property
import cirq
from cirq_qubitization.unary_iteration import UnaryIterationGate
from cirq_qubitization.gate_with_registers import Registers


class ApplyGateToLthQubit(UnaryIterationGate):
    r"""A controlled SELECT operation for single-qubit gates.

    $$
    \mathrm{SELECT} = \sum_{l}|l \rangle \langle l| \otimes [G(l)]_l
    $$

    Where $G$ is a function that maps an index to a single-qubit gate.

    This gate uses the unary iteration scheme to apply `nth_gate(selection)` to the
    `selection`-th qubit of `target` all controlled by the `control` register.

    Args:
        selection_bitsize: The size of the indexing `select` register. This should be at most
            `log2(target_bitsize)`
        target_bitsize: The size of the `target` register. This also serves as the iteration
            length.
        nth_gate: A function mapping the selection index to a single-qubit gate.
        control_bitsize: The size of the control register.
    """

    def __init__(
        self,
        selection_bitsize: int,
        target_bitsize: int,
        nth_gate: Callable[[int], cirq.Gate],
        *,
        control_bitsize: int = 1,
    ):
        self._selection_bitsize = selection_bitsize
        self._target_bitsize = target_bitsize
        self._nth_gate = nth_gate
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
