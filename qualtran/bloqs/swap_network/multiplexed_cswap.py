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
from typing import Tuple

import cirq
from attrs import field, frozen

from qualtran import bloq_example, BloqDocSpec, QAny, Register
from qualtran._infra.gate_with_registers import total_bits
from qualtran.bloqs.basic_gates import CSwap
from qualtran.bloqs.multiplexers.unary_iteration_bloq import UnaryIterationGate


@frozen
class MultiplexedCSwap(UnaryIterationGate):
    r"""Swaps the $l$-th register into an ancilla using unary iteration.

    Applies the unitary which performs
    $$
        U |l\rangle|\psi_0\rangle\cdots|\psi_l\rangle\cdots|\psi_n\rangle|\mathrm{junk}\rangle
        \rightarrow
        |l\rangle|\psi_0\rangle\cdots|\mathrm{junk}\rangle\cdots|\psi_n\rangle|\psi_l\rangle
    $$
    through a combination of unary iteration and CSwaps.

    The toffoli cost should be $L n_b + L - 2 + n_c$, where $L$ is the
    iteration length, $n_b$ is the bitsize of
    the registers to swap, and $n_c$ is the number of controls.

    Args:
        selection_regs: Indexing `select` signature of type Tuple[`Register`, ...].
            It also contains information about the iteration length of each selection register.
        target_bitsize: The size of the registers we want to swap.
        control_regs: Control registers for constructing a controlled version of the gate.

    Registers:
        control_registers: Control registers
        selection_regs: Indexing `select` signature of type Tuple[`Register`, ...].
            It also contains information about the iteration length of each selection register.
        target_registers: Target registers to swap. We swap FROM registers
            labelled x`i`, where i is an integer and TO a single register called y

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](https://arxiv.org/abs/2105.12767)
        page 20 paragraph 2.
    """
    selection_regs: Tuple[Register, ...] = field(
        converter=lambda v: (v,) if isinstance(v, Register) else tuple(v)
    )
    target_bitsize: int
    control_regs: Tuple[Register, ...] = field(
        converter=lambda v: (v,) if isinstance(v, Register) else tuple(v), default=()
    )

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return self.control_regs

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        return self.selection_regs

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        target_shape = tuple(
            sreg.dtype.iteration_length_or_zero() for sreg in self.selection_registers
        )
        return tuple(
            [
                Register('targets', QAny(bitsize=self.target_bitsize), shape=target_shape),
                Register('output', QAny(bitsize=self.target_bitsize)),
            ]
        )

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["@"] * total_bits(self.control_registers)
        wire_symbols += ["In"] * total_bits(self.selection_registers)
        for i, target in enumerate(self.target_registers):
            if i == len(self.target_registers) - 1:
                wire_symbols += ["×(y)"] * target.total_bits()
            else:
                wire_symbols += ["×(x)"] * target.total_bits()
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def nth_operation(
        self, context: cirq.DecompositionContext, control: cirq.Qid, **kwargs
    ) -> cirq.OP_TREE:
        selection_idx = tuple(kwargs[reg.name] for reg in self.selection_registers)
        target_regs = kwargs['targets']
        output_reg = kwargs['output']
        return CSwap(self.target_bitsize).make_on(
            ctrl=[control], x=target_regs[selection_idx], y=output_reg
        )


@bloq_example
def _multiplexed_cswap() -> MultiplexedCSwap:
    from qualtran import BQUInt

    selection_bitsize = 3
    iteration_length = 5
    target_bitsize = 2
    multiplexed_cswap = MultiplexedCSwap(
        Register('selection', BQUInt(selection_bitsize, iteration_length)),
        target_bitsize=target_bitsize,
    )

    return multiplexed_cswap


_MULTIPLEXED_CSWAP_DOC = BloqDocSpec(bloq_cls=MultiplexedCSwap, examples=(_multiplexed_cswap,))
