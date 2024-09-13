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

import itertools
from functools import cached_property
from typing import Callable, Sequence, Tuple

import attrs
import cirq
import numpy as np
import sympy

from qualtran import bloq_example, BloqDocSpec, BQUInt, QAny, QBit, Register, Signature
from qualtran._infra.gate_with_registers import total_bits
from qualtran.bloqs.multiplexers.unary_iteration_bloq import UnaryIterationGate


@attrs.frozen
class ApplyGateToLthQubit(UnaryIterationGate):
    r"""A controlled SELECT operation for single-qubit gates.

    $$
    \mathrm{SELECT} = \sum_{l}|l \rangle \langle l| \otimes [G(l)]_l
    $$

    Where $G$ is a function that maps an index to a single-qubit gate.

    This gate uses the unary iteration scheme to apply `nth_gate(selection)` to the
    `selection`-th qubit of `target` all controlled by the `control` register.

    Args:
        selection_regs: Indexing `select` signature of type Tuple[`Register`, ...].
            It also contains information about the iteration length of each selection register.
        nth_gate: A function mapping the composite selection index to a single-qubit gate.
        control_regs: Control signature for constructing a controlled version of the gate.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
        Babbush et. al. (2018). Section III.A. and Figure 7.
    """
    selection_regs: Tuple[Register, ...] = attrs.field(
        converter=lambda v: (v,) if isinstance(v, Register) else tuple(v)
    )
    nth_gate: Callable[..., cirq.Gate]
    control_regs: Tuple[Register, ...] = attrs.field(
        converter=lambda v: (v,) if isinstance(v, Register) else tuple(v),
        default=(Register('control', QBit()),),
    )

    @classmethod
    def make_on(
        cls, *, nth_gate: Callable[..., cirq.Gate], **quregs: Sequence[cirq.Qid]
    ) -> cirq.Operation:
        """Helper constructor to automatically deduce bitsize attributes."""
        return ApplyGateToLthQubit(
            Register('selection', BQUInt(len(quregs['selection']), len(quregs['target']))),
            nth_gate=nth_gate,
            control_regs=Register('control', QAny(len(quregs['control']))),
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
        total_iteration_size = np.prod(
            tuple(int(reg.dtype.iteration_length_or_zero()) for reg in self.selection_registers)
        )
        return (Register('target', QAny(int(total_iteration_size))),)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        if any(
            isinstance(reg.dtype.iteration_length_or_zero(), sympy.Expr)
            for reg in self.selection_registers
        ):
            raise ValueError(f'Symbolic iteration size not allowed for {self.selection_registers}')
        wire_symbols = ["@"] * total_bits(self.control_registers)
        wire_symbols += ["In"] * total_bits(self.selection_registers)
        for it in itertools.product(
            *[range(int(reg.dtype.iteration_length_or_zero())) for reg in self.selection_regs]
        ):
            wire_symbols += [str(self.nth_gate(*it))]
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def nth_operation(  # type: ignore[override]
        self,
        context: cirq.DecompositionContext,
        control: cirq.Qid,
        target: Sequence[cirq.Qid],
        **selection_indices: int,
    ) -> cirq.OP_TREE:
        if any(
            isinstance(reg.dtype.iteration_length_or_zero(), sympy.Expr)
            for reg in self.selection_registers
        ):
            raise ValueError(f'Symbolic iteration size not allowed for {self.selection_registers}')
        selection_shape = tuple(
            int(reg.dtype.iteration_length_or_zero()) for reg in self.selection_regs
        )
        selection_idx = tuple(selection_indices[reg.name] for reg in self.selection_regs)
        target_idx = int(np.ravel_multi_index(selection_idx, selection_shape))
        return self.nth_gate(*selection_idx).on(target[target_idx]).controlled_by(control)


@bloq_example
def _apply_z_to_odd() -> ApplyGateToLthQubit:
    from qualtran import BQUInt, Register

    def _z_to_odd(n: int):
        if n % 2 == 1:
            return cirq.Z
        return cirq.I

    apply_z_to_odd = ApplyGateToLthQubit(
        Register('selection', BQUInt(3, 4)),
        nth_gate=_z_to_odd,
        control_regs=Signature.build(control=2),
    )

    return apply_z_to_odd


_APPLY_TO_LTH_TARGET_DOC = BloqDocSpec(
    bloq_cls=ApplyGateToLthQubit,
    import_line='from qualtran.bloqs.multiplexers.apply_gate_to_lth_target import ApplyGateToLthQubit',
    examples=(_apply_z_to_odd,),
)
