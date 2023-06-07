"""Gates for applying generic selected unitaries."""
from functools import cached_property
from typing import Collection, List, Optional, Sequence, Tuple, Union

import cirq
import numpy as np

from cirq_qubitization.cirq_algos.unary_iteration import UnaryIterationGate
from cirq_qubitization.cirq_infra.gate_with_registers import Register, Registers, SelectionRegisters


@cirq.value_equality()
class GenericSelect(UnaryIterationGate):
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
        *,
        control_val: Optional[int] = None,
    ):
        if any(len(dps) != target_bitsize for dps in select_unitaries):
            raise ValueError(
                f"Each dense pauli string in `select_unitaries` should contain "
                f"{target_bitsize} terms."
            )
        self._selection_bitsize = selection_bitsize
        self._target_bitsize = target_bitsize
        self.select_unitaries = tuple(select_unitaries)
        self._control_val = control_val
        if self._selection_bitsize < (len(select_unitaries) - 1).bit_length():
            raise ValueError("Input selection_bitsize is not consistent with select_unitaries")

    @cached_property
    def control_registers(self) -> Registers:
        registers = [] if self._control_val is None else [Register('control', 1)]
        return Registers(registers)

    @cached_property
    def selection_registers(self) -> Registers:
        return SelectionRegisters.build(
            selection=(self._selection_bitsize, len(self.select_unitaries))
        )

    @cached_property
    def target_registers(self) -> Registers:
        return Registers.build(target=self._target_bitsize)

    def decompose_from_registers(self, context, **qubit_regs: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        if self._control_val == 0:
            yield cirq.X(*qubit_regs['control'])
        yield from super().decompose_from_registers(context, **qubit_regs)
        if self._control_val == 0:
            yield cirq.X(*qubit_regs['control'])

    def nth_operation(
        self,
        context: cirq.DecompositionContext,
        selection: int,
        control: cirq.Qid,
        target: Sequence[cirq.Qid],
    ) -> cirq.OP_TREE:
        """Applies `self.select_unitaries[selection]`.

        Args:
             selection: takes on values [0, self.iteration_lengths[0])
             control: Qid that is the control qubit or qubits
             target: Target register qubits
        """
        if selection < 0 or selection >= 2**self._selection_bitsize:
            raise ValueError("n is outside selection length range")
        ps = self.select_unitaries[selection].on(*target)
        return ps.with_coefficient(np.sign(ps.coefficient.real)).controlled_by(control)

    def controlled(
        self,
        num_controls: int = None,
        control_values: Sequence[Union[int, Collection[int]]] = None,
        control_qid_shape: Optional[Tuple[int, ...]] = None,
    ) -> 'GenericSelect':
        if num_controls is None:
            num_controls = 1
        if control_values is None:
            control_values = [1] * num_controls
        if len(control_values) == 1 and self._control_val is None:
            return GenericSelect(
                self._selection_bitsize,
                self._target_bitsize,
                self.select_unitaries,
                control_val=control_values[0],
            )
        raise NotImplementedError(f'Cannot create a controlled version of {self}')

    def _value_equality_values_(self):
        return (
            self.select_unitaries,
            self._selection_bitsize,
            self._target_bitsize,
            self._control_val,
        )


GenericSelect.__hash__ = cirq._compat.cached_method(GenericSelect.__hash__)
