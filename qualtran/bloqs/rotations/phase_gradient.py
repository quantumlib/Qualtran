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

import attrs
import cirq
from numpy.typing import NDArray

from qualtran import GateWithRegisters, Register, Side, Signature
from qualtran.bloqs.basic_gates import Hadamard
from qualtran.bloqs.basic_gates.rotation import CZPowGate, ZPowGate
from qualtran.bloqs.on_each import OnEach


@attrs.frozen
class PhaseGradientSchoolBook(GateWithRegisters):
    r"""A naive implementation (Controlled-/)PhaseGradient gate on an n-bit register.

    Supports both
    $$
        \text{PhaseGrad}_{n} = \sum_{k=0}^{2^{n}-1}|k\rangle\ langle k| \omega_{n, t}^{k}
    $$
    and
    $$
        \text{CPhaseGrad}_{n} = \sum_{k=0}^{2^{n}-1} |0\rangle \langle 0| I + |1\rangle \langle1| \omega_{n, t}^{k}$
    $$
    where
    $$
        \omega_{n, t} = \exp\left(\frac{2\pi i t}{2^n}\right)
    $$
    """
    bitsize: int
    exponent: int = 1
    controlled: bool = False
    eps: float = 1e-10

    @cached_property
    def signature(self) -> 'Signature':
        return (
            Signature.build(ctrl=1, phase_grad=self.bitsize)
            if self.controlled
            else Signature.build(phase_grad=self.bitsize)
        )

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        ctrl = quregs.get('ctrl', ())
        gate = CZPowGate if self.controlled else ZPowGate
        for i, q in enumerate(quregs['phase_grad']):
            yield gate(exponent=self.exponent / 2**i, eps=self.eps / self.bitsize).on(*ctrl, q)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbols = ['@'] * self.controlled + [f'Z^1/{2**(i+1)}' for i in range(self.bitsize)]
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def __pow__(self, power):
        if power == 1:
            return self
        return PhaseGradientSchoolBook(
            self.bitsize, self.exponent * power, self.controlled, self.eps
        )


@attrs.frozen
class PhaseGradientState(GateWithRegisters):
    r"""Prepare a phase gradient state $|\phi\rangle$ on a new register of bitsize $b_{grad}$

    $$
        |\phi\rangle = \frac{1}{\sqrt{2^{n}}} \sum_{k=0}^{2^{n} - 1} \omega_{n}^{-k} |k\rangle
    $$

    where

    $$
        \omega_{n} = \exp\left(\frac{2\pi i}{2^n}\right)
    $$

    References:
        [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization]
        (https://arxiv.org/abs/2007.07391)
        Appendix A: Addition for controlled rotations
    """

    bitsize: int
    adjoint: bool = False
    eps: float = 1e-10

    @cached_property
    def signature(self) -> 'Signature':
        side = Side.LEFT if self.adjoint else Side.RIGHT
        return Signature([Register('phase_grad', self.bitsize, side=side)])

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        # Assumes `phase_grad` is in big-endian representation.
        phase_grad = quregs['phase_grad']
        ops = [OnEach(self.bitsize, Hadamard()).on_registers(q=phase_grad)]
        ops += [
            PhaseGradientSchoolBook(self.bitsize, exponent=-1).on_registers(phase_grad=phase_grad)
        ]
        yield cirq.inverse(ops) if self.adjoint else ops

    def __pow__(self, power):
        if power == 1:
            return self
        if power == -1:
            return PhaseGradientState(self.bitsize, self.adjoint ^ True, self.eps)
        raise NotImplementedError(f"Power is only defined for +1/-1. Found {self.power}.")
