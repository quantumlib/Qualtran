"""Gates for applying generic selected unitaries."""
from functools import cached_property
from typing import List, Sequence, Tuple

import cirq

from cirq_qubitization import unary_iteration
from cirq_qubitization.cirq_infra.gate_with_registers import Registers


class GenericSelect(unary_iteration.UnaryIterationGate):
    r"""A SELECT gate for selecting and applying operators from an array of `PauliString`s.

    $$
    \mathrm{SELECT} = \sum_{l}|l \rangle \langle l| \otimes U_l
    $$

    Where $U_l$ is a member of the Pauli group.

    This gate uses the unary iteration scheme to apply `select_unitaries[selection]` to `target`
    controlled on the single-bit `control` register.

    Args:
        selection_bitsize: The size of the indexing `select` register. This should be at least
            `log2(len(select_unitaries))`
        target_bitsize: The size of the `target` register.
        select_unitaries: List of `DensePauliString`s to apply to the `target` register. Each
            dense pauli string must contain `target_bitsize` terms.
    """

    def __init__(
        self,
        selection_bitsize: int,
        target_bitsize: int,
        select_unitaries: List[cirq.DensePauliString],
    ):
        if any(len(dps) != target_bitsize for dps in select_unitaries):
            raise ValueError(
                f"Each dense pauli string in `select_unitaries` should contain "
                f"{target_bitsize} terms."
            )
        self._selection_bitsize = selection_bitsize
        self._target_bitsize = target_bitsize
        self.select_unitaries = select_unitaries
        if self._selection_bitsize < (len(select_unitaries) - 1).bit_length():
            raise ValueError("Input selection_bitsize is not consistent with select_unitaries")

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
        return (len(self.select_unitaries),)

    def nth_operation(
        self, selection: int, control: cirq.Qid, target: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        """Applies `self.select_unitaries[selection]`.

        Args:
             selection: takes on values [0, self.iteration_lengths[0])
             control: Qid that is the control qubit or qubits
             target: Target register qubits
        """
        if selection < 0 or selection >= 2**self._selection_bitsize:
            raise ValueError("n is outside selection length range")
        return (
            self.select_unitaries[selection].on(*target).with_coefficient(1).controlled_by(control)
        )
