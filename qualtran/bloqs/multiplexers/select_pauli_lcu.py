#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Bloqs for applying SELECT unitary for LCU of Pauli Strings."""

from functools import cached_property
from typing import Collection, Optional, Sequence, Tuple, Union

import attrs
import cirq
import numpy as np
from numpy.typing import NDArray

from qualtran import bloq_example, BloqDocSpec, BoundedQUInt, QAny, QBit, Register
from qualtran.bloqs.multiplexers.unary_iteration_bloq import UnaryIterationGate
from qualtran.bloqs.select_and_prepare import SelectOracle
from qualtran.resource_counting.generalizers import (
    cirq_to_bloqs,
    ignore_cliffords,
    ignore_split_join,
)


@attrs.frozen
class SelectPauliLCU(SelectOracle, UnaryIterationGate):
    r"""A SELECT bloq for selecting and applying operators from an array of `PauliString`s.

    $$
    \mathrm{SELECT} = \sum_{l}|l \rangle \langle l| \otimes U_l
    $$

    Where $U_l$ is a member of the Pauli group.

    This bloq uses the unary iteration scheme to apply `select_unitaries[selection]` to `target`
    controlled on the single-bit `control` register.

    Args:
        selection_bitsize: The size of the indexing `select` register. This should be at least
            `log2(len(select_unitaries))`
        target_bitsize: The size of the `target` register.
        select_unitaries: List of `DensePauliString`s to apply to the `target` register. Each
            dense pauli string must contain `target_bitsize` terms.
        control_val: Optional control value. If specified, a singly controlled gate is constructed.
    """
    selection_bitsize: int
    target_bitsize: int
    select_unitaries: Tuple[cirq.DensePauliString, ...] = attrs.field(converter=tuple)
    control_val: Optional[int] = None

    def pretty_name(self) -> str:
        return 'SelectPauliLCU'

    def __attrs_post_init__(self):
        if any(len(dps) != self.target_bitsize for dps in self.select_unitaries):
            raise ValueError(
                f"Each dense pauli string in {self.select_unitaries} should contain "
                f"{self.target_bitsize} terms."
            )
        min_bitsize = (len(self.select_unitaries) - 1).bit_length()
        if self.selection_bitsize < min_bitsize:
            raise ValueError(
                f"selection_bitsize={self.selection_bitsize} should be at-least {min_bitsize}"
            )

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return () if self.control_val is None else (Register('control', QBit()),)

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        return (
            Register('selection', BoundedQUInt(self.selection_bitsize, len(self.select_unitaries))),
        )

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return (Register('target', QAny(self.target_bitsize)),)

    def decompose_from_registers(
        self, context, **quregs: NDArray[cirq.Qid]  # type:ignore[type-var]
    ) -> cirq.OP_TREE:
        if self.control_val == 0:
            yield cirq.X(*quregs['control'])
        yield super(SelectPauliLCU, self).decompose_from_registers(context=context, **quregs)
        if self.control_val == 0:
            yield cirq.X(*quregs['control'])

    def nth_operation(  # type: ignore[override]
        self,
        context: cirq.DecompositionContext,
        selection: int,
        control: cirq.Qid,
        target: Sequence[cirq.Qid],
    ) -> cirq.OP_TREE:
        """Applies `self.select_unitaries[selection]`.

        Args:
             context: `cirq.DecompositionContext` stores options for decomposing gates (eg:
                cirq.QubitManager).
             selection: takes on values [0, self.iteration_lengths[0])
             control: Qid that is the control qubit or qubits
             target: Target register qubits
        """
        ps = self.select_unitaries[selection].on(*target)
        return ps.with_coefficient(np.sign(complex(ps.coefficient).real)).controlled_by(control)

    def controlled(
        self,
        num_controls: Optional[int] = None,
        control_values: Optional[
            Union[cirq.ops.AbstractControlValues, Sequence[Union[int, Collection[int]]]]
        ] = None,
        control_qid_shape: Optional[Tuple[int, ...]] = None,
    ) -> 'SelectPauliLCU':
        if num_controls is None:
            num_controls = 1
        if control_values is None:
            control_values = [1] * num_controls
        if (
            isinstance(control_values, Sequence)
            and isinstance(control_values[0], int)
            and len(control_values) == 1
            and self.control_val is None
        ):
            return SelectPauliLCU(
                self.selection_bitsize,
                self.target_bitsize,
                self.select_unitaries,
                control_val=control_values[0],
            )
        raise NotImplementedError(
            f'Cannot create a controlled version of {self} with control_values={control_values}.'
        )


@bloq_example(generalizer=[cirq_to_bloqs, ignore_split_join, ignore_cliffords])
def _select_pauli_lcu() -> SelectPauliLCU:
    target_bitsize = 4
    us = ['XIXI', 'YIYI', 'ZZZZ', 'ZXYZ']
    us = [cirq.DensePauliString(u) for u in us]
    selection_bitsize = int(np.ceil(np.log2(len(us))))
    select_pauli_lcu = SelectPauliLCU(selection_bitsize, target_bitsize, select_unitaries=us)
    return select_pauli_lcu


_SELECT_PAULI_LCU_DOC = BloqDocSpec(
    bloq_cls=SelectPauliLCU,
    import_line='from qualtran.bloqs.multiplexers.select_pauli_lcu import SelectPauliLCU',
    examples=(_select_pauli_lcu,),
)
