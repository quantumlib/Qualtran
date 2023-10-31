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
from typing import Tuple

import attr
import cirq
from cirq._compat import cached_property

from qualtran import Register, SelectionRegister
from qualtran._infra.gate_with_registers import total_bits
from qualtran.bloqs.basic_gates import CSwap
from qualtran.bloqs.unary_iteration_bloq import UnaryIterationGate


@attr.frozen
class ApplyCSwapToLthReg(UnaryIterationGate):
    r"""Swaps the $l$-th register into an ancilla using cswaps and unary iteration.

    Args:
        bitsize: The size of the registers we want to swap.
        selection_regs: Indexing `select` signature of type Tuple[`SelectionRegisters`, ...].
            It also contains information about the iteration length of each selection register.
        nth_gate: A function mapping the composite selection index to a single-qubit gate.

    References:
    """
    bitsize: int
    selection_regs: Tuple[SelectionRegister, ...] = attr.field(
        converter=lambda v: (v,) if isinstance(v, SelectionRegister) else tuple(v)
    )
    control_regs: Tuple[Register, ...] = attr.field(
        converter=lambda v: (v,) if isinstance(v, Register) else tuple(v),
        default=(Register('control', 0),),
    )

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return self.control_regs

    @cached_property
    def selection_registers(self) -> Tuple[SelectionRegister, ...]:
        return self.selection_regs

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        # only one selection register
        iteration_length = self.selection_registers[0].iteration_length
        regs = [Register(f'x{i}', bitsize=self.bitsize) for i in range(iteration_length)]
        regs += [Register('y', bitsize=self.bitsize)]
        return tuple(regs)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["@"] * total_bits(self.control_registers)
        wire_symbols += ["In"] * total_bits(self.selection_registers)
        for i, target in enumerate(self.target_registers):
            if i == len(self.target_registers) - 1:
                wire_symbols += [f"×(y)"] * target.total_bits()
            else:
                wire_symbols += [f"×(x)"] * target.total_bits()
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def nth_operation(
        self, context: cirq.DecompositionContext, control: cirq.Qid, **kwargs
    ) -> cirq.OP_TREE:
        selection_idx = tuple(kwargs[reg.name] for reg in self.selection_registers)[0]
        target_regs = {reg.name: kwargs[reg.name] for reg in self.target_registers}
        return CSwap(self.bitsize).make_on(
            ctrl=control, x=target_regs[f'x{selection_idx}'], y=target_regs['y']
        )
