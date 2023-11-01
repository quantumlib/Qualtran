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
    r"""Swaps $l$-th register into an ancilla using unary iteration.

    Applies the unitary which peforms
    $$
        U |l\rangle|\psi_0\rangle\cdots|\psi_l\rangle|\psi_n\rangle|\mathrm{junk}\rangle
        \rightarrow
        |l\rangle|\psi_0\rangle\cdots|\mathrm{junk}|\psi_n\rangle|\psi_l\rangle \rangle
    $$
    through a combination of unary iteration and CSwaps.

    The cost should be bitsize * iteration_length + iteration_length - 2 + num_controls

    Args:
        selection_regs: Indexing `select` signature of type Tuple[`SelectionRegisters`, ...].
            It also contains information about the iteration length of each selection register.
        bitsize: The size of the registers we want to swap.
        nth_gate: A function mapping the composite selection index to a single-qubit gate.

    Registers:
        control_registers: Control registers
        selection_regs: Indexing `select` signature of type Tuple[`SelectionRegisters`, ...].
            It also contains information about the iteration length of each selection register.
        target_registers: Target registers to swap. We swap FROM registers
            labelled x`i`, where i is an integer and TO a single register called y

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](
            https://arxiv.org/abs/2105.12767) page 20 paragraph 2.
    """
    selection_regs: Tuple[SelectionRegister, ...] = attr.field(
        converter=lambda v: (v,) if isinstance(v, SelectionRegister) else tuple(v)
    )
    bitsize: int
    control_regs: Tuple[Register, ...] = attr.field(
        converter=lambda v: (v,) if isinstance(v, Register) else tuple(v),
        default=(Register('ctrl', 0),),
    )

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return self.control_regs

    @cached_property
    def selection_registers(self) -> Tuple[SelectionRegister, ...]:
        return self.selection_regs

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
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
