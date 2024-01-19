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
from typing import Dict, Iterable, Sequence, Set, TYPE_CHECKING, Union

import attrs
import cirq
from numpy.typing import NDArray

from qualtran import GateWithRegisters, Register, Side, Signature
from qualtran.bloqs.basic_gates import Hadamard, Toffoli
from qualtran.bloqs.basic_gates.rotation import CZPowGate, ZPowGate
from qualtran.bloqs.on_each import OnEach
from qualtran.cirq_interop.bit_tools import float_as_fixed_width_int

if TYPE_CHECKING:
    from qualtran.resource_counting.bloq_counts import BloqCountT


@attrs.frozen
class PhaseGradientUnitary(GateWithRegisters):
    r"""Implementation of (Controlled-/)PhaseGradient unitary gate on an n-bit register.

    The class supports implementing the phase gradient unitary gate and a controlled version
    thereof. The n bit phase gradient unitary is defined as

    $$
        \text{PhaseGrad}_{n, t} = \sum_{k=0}^{2^{n}-1}|k\rangle\ langle k| \omega_{n, t}^{k}
    $$

    where

    $$
        \omega_{n, t} = \exp\left(\frac{2\pi i t}{2^n}\right)
    $$

    The implementation simply decomposes into $n$ (controlled-) rotations, one on each qubit.
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
        wire_symbols = ['@'] * self.controlled + [
            f'Z^{self.exponent}/{2**(i+1)}' for i in range(self.bitsize)
        ]
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def __pow__(self, power):
        if power == 1:
            return self
        return PhaseGradientUnitary(self.bitsize, self.exponent * power, self.controlled, self.eps)


@attrs.frozen
class PhaseGradientState(GateWithRegisters):
    r"""Prepare a phase gradient state $|\phi\rangle$ on a new register of bitsize $b_{grad}$

    $$
        |\phi\rangle = \frac{1}{\sqrt{2^{n}}} \sum_{k=0}^{2^{n} - 1} \omega_{n, t}^{k} |k\rangle
    $$

    where

    $$
        \omega_{n, t} = \exp\left(\frac{2\pi i t}{2^n}\right)
    $$

    Allocates / deallocates registers to store the phase gradient state and delegates
    to the `PhaseGradientUnitary` bloq defined above.

    References:
        [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization]
        (https://arxiv.org/abs/2007.07391)
        Appendix A: Addition for controlled rotations
    """

    bitsize: int
    exponent: int = -1
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
            PhaseGradientUnitary(self.bitsize, exponent=self.exponent).on_registers(
                phase_grad=phase_grad
            )
        ]
        yield cirq.inverse(ops) if self.adjoint else ops

    def __pow__(self, power):
        if power == 1:
            return self
        if power == -1:
            return PhaseGradientState(self.bitsize, self.exponent, not self.adjoint, self.eps)
        raise NotImplementedError(f"Power is only defined for +1/-1. Found {self.power}.")


@attrs.frozen
class AddIntoPhaseGrad(GateWithRegisters, cirq.ArithmeticGate):
    r"""Quantum-quantum addition into a phase gradient register using $b_{phase} - 2$ Toffolis

    $$
        U|x\rangle|\text{phase\_grad}\rangle = |x\rangle|\text{phase\_grad} + x\rangle
    $$

    Args:
        inp_bitsize: Size of input register.
        phase_bitsize: Size of phase gradient register to which the input value should be added.

    Registers:
        - x : Input THRU register storing input value x to be added to the phase gradient register.
        - phase_grad : Phase gradient THRU register.

    References:
        [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization]
        (https://arxiv.org/abs/2007.07391), Appendix A: Addition for controlled rotations
    """
    inp_bitsize: int
    phase_bitsize: int

    def __attrs_post_init__(self):
        assert self.phase_bitsize >= self.inp_bitsize

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(x=self.inp_bitsize, phase_grad=self.phase_bitsize)

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return [2] * self.inp_bitsize, [2] * self.phase_bitsize

    def with_registers(self, *new_registers: Union[int, Sequence[int]]):
        raise NotImplementedError("not needed.")

    def apply(self, x, phase_grad) -> Union[int, Iterable[int]]:
        return x, phase_grad + x

    def on_classical_vals(self, x, phase_grad) -> Dict[str, 'ClassicalValT']:
        return {'x': x, 'phase_grad': (phase_grad + x) % (2**self.phase_bitsize)}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        num_toffoli = self.phase_bitsize - 2
        return {(Toffoli(), num_toffoli)}

    def _t_complexity_(self) -> 'TComplexity':
        ((toffoli, n),) = self.bloq_counts().items()
        return n * toffoli.t_complexity()


@attrs.frozen
class AddScaledValIntoPhaseReg(GateWithRegisters, cirq.ArithmeticGate):
    r"""Optimized quantum-quantum addition into a phase gradient register scaled by a constant $\gamma$.

    $$
        U(\gamma)|x\rangle|\text{phase\_grad}\rangle = |x\rangle|\text{phase\_grad} + x * \gamma\rangle
    $$

    The operation calls `AddIntoPhaseGrad` gate $(gamma_bitsize + 2) / 2$ times.

    Args:
        inp_bitsize: Size of input register.
        phase_bitsize: Size of phase gradient register to which the scaled input should be added.
        gamma: Floating point scaling factor in the range [0, 1].
        gamma_bitsize: Number of bits of precisions to be used for `gamma`.

    Registers:
        - x : Input THRU register storing input value x to be scaled and added to the phase
            gradient register.
        - phase_grad : Phase gradient THRU register.

    References:
        [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization]
        (https://arxiv.org/abs/2007.07391), Appendix A: Addition for controlled rotations
    """

    inp_bitsize: int
    phase_bitsize: int
    gamma: float
    gamma_bitsize: int

    def __attrs_post_init__(self):
        assert 0 <= self.gamma <= 1

    @cached_property
    def signature(self):
        return Signature.build(x=self.inp_bitsize, phase_grad=self.phase_bitsize)

    def registers(self):
        return [2] * self.inp_bitsize, [2] * self.phase_bitsize

    def with_registers(self, *new_registers: Union[int, Sequence[int]]):
        raise NotImplementedError("not needed.")

    @cached_property
    def gamma_int_numerator(self) -> int:
        _, gamma_fixed_width_int = float_as_fixed_width_int(self.gamma, self.gamma_bitsize + 1)
        gamma_fixed_width_float = gamma_fixed_width_int / 2**self.gamma_bitsize
        _, gamma_numerator = float_as_fixed_width_int(
            gamma_fixed_width_float, self.phase_bitsize + 1
        )
        return gamma_numerator

    def apply(self, x: int, phase_grad: int) -> Union[int, Iterable[int]]:
        return x, phase_grad + self.gamma_int_numerator * x

    def on_classical_vals(self, x, phase_grad) -> Dict[str, 'ClassicalValT']:
        phase_grad_out = (phase_grad + self.gamma_int_numerator * x) % 2**self.phase_bitsize
        return {'x': x, 'phase_grad': phase_grad_out}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        num_additions = (self.gamma_bitsize + 2) // 2
        return {(AddIntoPhaseGrad(self.inp_bitsize, self.phase_bitsize), num_additions)}

    def _t_complexity_(self):
        ((add_into_phase, n),) = self.bloq_counts().items()
        return n * add_into_phase.t_complexity()
