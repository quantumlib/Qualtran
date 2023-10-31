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

from typing import Dict, Sequence, Set, TYPE_CHECKING, Union

import attrs
import cirq
from numpy.typing import NDArray
from functools import cached_property

from qualtran import GateWithRegisters, Register, Side, Signature
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.bloqs.basic_gates.rotation import ZPowGate
from qualtran.cirq_interop.t_complexity_protocol import TComplexity

if TYPE_CHECKING:
    from qualtran.resource_counting import SympySymbolAllocator, BloqCountT
    from qualtran.simulation.classical_sim import ClassicalValT


@attrs.frozen
class PhaseGradient(GateWithRegisters):
    r"""Phases all computational basis states proportional to the integer value of the state.

    $$
        \mathrm{Grad}_{n}^{t} = \sum_{k=0}^{2^{n}-1}|k\rangle\ langle k| \omega_{n, t}^{k}
    $$

    where
    $$
        \omega_{n, t} = \exp\left(\frac{2\pi t i}{2^n}\right)
    $$

    References:
        [Efficient Controlled Phase Gradients](https://algassert.com/post/1708)
    """
    bitsize: int
    exponent: float = 1
    eps: float = 1e-10

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(phase_reg=self.bitsize)

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        for i, q in enumerate(quregs['phase_reg']):
            yield ZPowGate(exponent=self.exponent / 2**i, eps=self.eps / self.bitsize).on(q)

    def pretty_name(self) -> str:
        exponent = f'^{self.exponent}' if self.exponent != 1 else ''
        return f'GRAD[{self.bitsize}]{exponent}'

    def __pow__(self, power):
        return PhaseGradient(self.bitsize, self.exponent**power, self.eps)


@attrs.frozen
class PhaseGradientState(GateWithRegisters):
    """Prepare a phase gradient state $|\phi\rangle$ on a new register of bitsize $b_{grad}$

    $$
        |\phi\rangle = \frac{1}{\sqrt{2^{n}}} \sum_{k=0}^{2^{n} - 1} \omega_{n}^{-k} |k\rangle

    $$

    where

    $$
        \omega_{n} = \exp\left(\frac{2\pi i}{2^n}\right)
    $$

    References:
        [](https://arxiv.org/abs/2007.07391)
        Appendix A: Addition for controlled rotations
    """

    b_grad: int
    adjoint: bool = False
    eps: float = 1e-10

    @cached_property
    def signature(self) -> 'Signature':
        side = Side.LEFT if self.adjoint else Side.RIGHT
        return Signature([Register('b_grad', self.b_grad, side=side)])

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        q = quregs['b_grad']
        if self.adjoint:
            yield PhaseGradient(bitsize=self.b_grad, exponent=+1, eps=self.eps).on_registers(
                phase_reg=q
            )
            yield cirq.H.on_each(*q)
        else:
            yield cirq.H.on_each(*q)
            yield PhaseGradient(bitsize=self.b_grad, exponent=-1, eps=self.eps).on_registers(
                phase_reg=q
            )

    def __pow__(self, power):
        if power == 1:
            return self
        if power == -1:
            return PhaseGradientState(self.b_grad, self.adjoint ^ True, self.eps)
        raise NotImplementedError(f"Power is only defined for +1/-1. Found {self.power}.")


@attrs.frozen
class AddIntoPhaseReg(GateWithRegisters):
    r"""Optimized quantum-quantum addition into a phase gradient register."""
    a_bitsize: int
    b_bitsize: int

    def __attrs_post_init__(self):
        assert self.b_bitsize >= self.a_bitsize

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(a=self.a_bitsize, b=self.b_bitsize)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        num_toffoli = self.b_bitsize - 2
        return {(Toffoli(), num_toffoli)}


@attrs.frozen
class AddScaledValIntoPhaseReg(GateWithRegisters):
    r"""Optimized quantum-quantum addition into a phase gradient register scaled by a constant $\gamma$."""
    gamma_bitsize: int
    input_bitsize: int
    phase_gradient_bitsize: int

    @cached_property
    def signature(self):
        return Signature.build(direct_reg=self.input_bitsize, phase_reg=self.phase_gradient_bitsize)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        num_additions = (self.gamma_bitsize + 2) // 2
        return {(AddIntoPhaseReg(self.input_bitsize, self.phase_gradient_bitsize), num_additions)}


@attrs.frozen
class AddConstantIntoPhaseReg(GateWithRegisters, cirq.ArithmeticGate):
    r"""Optimized classical-quantum addition into a phase gradient register."""

    bitsize: int
    add_val: int

    @cached_property
    def signature(self):
        return Signature.build(phase_reg=self.bitsize)

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return [2] * self.bitsize, self.add_val

    def apply(self, *register_vals: int) -> Union[int, Sequence[int]]:
        phase_reg_val, add_val = register_vals
        return phase_reg_val + add_val, add_val

    def on_classical_vals(self, phase_reg: 'ClassicalValT') -> Dict[str, 'ClassicalValT']:
        return {'phase_reg': (phase_reg + self.add_val) % 2**self.bitsize}

    def t_complexity(self):
        num_clifford = (self.bitsize - 3) * 19 + 16
        num_t_gates = 4 * (self.bitsize - 3)
        return TComplexity(t=num_t_gates, clifford=num_clifford)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        num_toffoli = self.bitsize - 3
        return {(Toffoli(), num_toffoli)}


@attrs.frozen
class CPhaseGradientViaAddition(GateWithRegisters):
    r"""Applies $e^{\frac{-2 \pi i k Z}{2^n}}$  on n-bit target register using phase gradient state.

    $$
        \mathrm{CGrad}_{n} = \sum_{k=0}^{2^n - 1} \left(
                                    \omega_{n}^{-k} |0\rangle \langle 0| +
                                    \omega_{n}^{k} |1\rangle \langle 1|
                                    \right)
                                    |k\rangle \langle k|
                                    |\phi\rangle \langle\phi|
    $$

    where

    $$
        \omega_{n} = \exp\left(\frac{2\pi i}{2^n}\right)
    $$

    References:
        [Efficient Controlled Phase Gradients](https://algassert.com/post/1708)
    """

    bitsize: int
    exponent: float = 1

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(ctrl=1, phase_reg=self.bitsize, resource_state=self.bitsize)

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        ctrl, x, phi = quregs['ctrl'], quregs['phase_reg'], quregs['resource_state']
        yield cirq.X(*ctrl)
        yield [cirq.CNOT(ctrl, t) for t in phi]
        yield AddIntoPhaseReg(self.bitsize).on_registers(a=x, b=phi)
        yield [cirq.CNOT(ctrl, t) for t in phi]
        yield cirq.X(*ctrl)
