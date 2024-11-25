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

from functools import cached_property
from typing import Iterator, Sequence, Tuple, Union

import attrs
import cirq
import numpy as np
import sympy
from numpy.typing import NDArray

from qualtran import QAny, QBit, Register
from qualtran._infra.data_types import BQUInt
from qualtran._infra.gate_with_registers import total_bits
from qualtran.bloqs.multiplexers.unary_iteration_bloq import UnaryIterationGate


@attrs.frozen
class SelectedMajoranaFermion(UnaryIterationGate):
    """Implements U s.t. U|l>|Psi> -> |l> T_{l} . Z_{l - 1} ... Z_{0} |Psi>

    where T is a single qubit target gate that defaults to pauli Y. The gate is
    implemented using an accumulator bit in the unary iteration circuit as explained
    in the reference below.


    Args:
        selection_regs: Indexing `select` signature of type `Register(dtype=BQUInt)`.
            It also contains information about the iteration length of each selection register.
        control_regs: Control signature for constructing a controlled version of the gate.
        target_gate: Single qubit gate to be applied to the target qubits.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
        Fig 9.
    """

    selection_regs: Tuple[Register, ...] = attrs.field(
        converter=lambda v: (v,) if isinstance(v, Register) else tuple(v)
    )
    control_regs: Tuple[Register, ...] = attrs.field(
        converter=lambda v: (v,) if isinstance(v, Register) else tuple(v),
        default=(Register('control', QBit()),),
    )
    target_gate: cirq.Gate = cirq.Y

    @classmethod
    def make_on(
        cls,
        *,
        target_gate=cirq.Y,
        **quregs: Union[Sequence[cirq.Qid], NDArray[cirq.Qid]],  # type: ignore[type-var]
    ) -> cirq.Operation:
        """Helper constructor to automatically deduce selection_regs attribute."""
        return SelectedMajoranaFermion(
            selection_regs=Register(
                'selection', BQUInt(len(quregs['selection']), len(quregs['target']))
            ),
            target_gate=target_gate,
        ).on_registers(**quregs)

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return self.control_regs

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        return self.selection_regs

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        if any(
            isinstance(reg.dtype.iteration_length_or_zero(), sympy.Expr)
            for reg in self.selection_registers
        ):
            raise ValueError(f'Symbolic iteration size not allowed for {self.selection_registers}')
        iteration_sizes = list(
            int(reg.dtype.iteration_length_or_zero()) for reg in self.selection_registers
        )
        total_iteration_size = np.prod(iteration_sizes)
        return (Register('target', QAny(int(total_iteration_size))),)

    @cached_property
    def extra_registers(self) -> Tuple[Register, ...]:
        return (Register('accumulator', QBit()),)

    def decompose_from_registers(
        self, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> Iterator[cirq.OP_TREE]:
        quregs['accumulator'] = np.array(context.qubit_manager.qalloc(1))
        control = quregs[self.control_regs[0].name] if total_bits(self.control_registers) else []
        yield cirq.X(*quregs['accumulator']).controlled_by(*control)
        yield super(SelectedMajoranaFermion, self).decompose_from_registers(
            context=context, **quregs
        )
        context.qubit_manager.qfree(quregs['accumulator'])

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["@"] * total_bits(self.control_registers)
        wire_symbols += ["In"] * total_bits(self.selection_registers)
        wire_symbols += [f"Z{self.target_gate}"] * total_bits(self.target_registers)
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def nth_operation(  # type: ignore[override]
        self,
        context: cirq.DecompositionContext,
        control: cirq.Qid,
        target: Sequence[cirq.Qid],
        accumulator: Sequence[cirq.Qid],
        **selection_indices: int,
    ) -> Iterator[cirq.OP_TREE]:
        if any(
            isinstance(reg.dtype.iteration_length_or_zero(), sympy.Expr)
            for reg in self.selection_regs
        ):
            raise ValueError(f'Symbolic iteration length not allowed for {self.selection_regs}')
        selection_shape = tuple(
            int(reg.dtype.iteration_length_or_zero()) for reg in self.selection_regs
        )
        selection_idx = tuple(selection_indices[reg.name] for reg in self.selection_regs)
        target_idx = int(np.ravel_multi_index(selection_idx, selection_shape))
        yield cirq.CNOT(control, *accumulator)
        yield self.target_gate(target[target_idx]).controlled_by(control)
        yield cirq.CZ(*accumulator, target[target_idx])

    def __str__(self):
        return f'SelectedMajoranaFermion({self.target_gate})'
