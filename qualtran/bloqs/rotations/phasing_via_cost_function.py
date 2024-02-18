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
from typing import Dict, Set, TYPE_CHECKING, Union

import attrs
import sympy

from qualtran import Bloq, BloqBuilder, GateWithRegisters, QFxp, Register, Signature
from qualtran.bloqs import utils
from qualtran.bloqs.basic_gates.rotation import ZPowGate
from qualtran.bloqs.rotations.phase_gradient import AddScaledValIntoPhaseReg

if TYPE_CHECKING:
    from qualtran import SoquetT


@attrs.frozen
class PhasingViaCostFunction(Bloq):
    r"""Phases every basis state $|x\rangle$ by an amount proportional to a cost function $f(x)$

    This Bloq implements a unitary $U_f(\gamma)$ which phases each computational state on which
    the wave-function has support, by an amount proportional to a function of that computational
    basis state. The general unitary can be defined as
    $$
        U_f(\gamma) = \sum_{x=0}^{N-1} e^{i 2 \pi \gamma f(x)} |x\rangle \langle x|
    $$

    The strategy to implement $U_f(\gamma)$ is to use two oracles $O_\text{direct}$
    and $O_\text{phase}$ s.t.
    $$
    U_f(\gamma) = O_\text{direct}^\dagger(\mathbb{I}\otimes O_\text{phase})O_\text{direct}
    $$

    $O^\text{direct}$ evaluates a $b_\text{direct}$-bit approximation of the cost function $f(x)$
    and stores it in a new output cost register. Note that the cost register can represent
    arbitrary fixed point values and be of type `QFxp(b_direct, n_frac, signed)`.
    $$
    O^\text{direct}|x\rangle|0\rangle^{\otimes b_\text{direct}}_\text{cost}=|x\rangle|f(x)\rangle
    $$

    $O^\text{phase}$ acts on the cost register computed by $O^\text{direct}$ and phases the
    state $|f(x)\rangle$ by $e^{i 2\pi \gamma f(x)}$
    $$
    O^\text{phase}(\gamma)=\sum_{k=0}^{2^{b_\text{direct}}-1}e^{i 2\pi\gamma k}|k\rangle\langle k|
    $$


    Different strategies for implementing the two oracles would give different costs tradeoffs.
    See `PhaseOracleZPow` and `PhaseOraclePhaseGradient` for two different implementations of
    phase oracles described in the reference.

    Args:
        cost_eval_oracle: Cost function evaluation oracle. Must compute the cost in a
            newly allocated RIGHT register.
        phase_oracle: Oracle to phase the cost register. Must consume the cost register
            allocated by `cost_eval_oracle` as a THRU input.

    References:
        [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization]
        (https://arxiv.org/abs/2007.07391), Appendix C: Oracles for phasing by cost function
    """
    cost_eval_oracle: Bloq
    phase_oracle: Bloq

    @cached_property
    def signature(self) -> 'Signature':
        registers = [*self.cost_eval_oracle.signature.lefts()]
        for reg in self.phase_oracle.signature.lefts():
            try:
                _ = self.cost_eval_oracle.signature.get_right(reg.name)
            except KeyError:
                registers.append(reg)
        registers = list(dict.fromkeys(registers).keys())
        return Signature(registers)

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        def _extract_soqs(bloq: Bloq) -> Dict[str, 'SoquetT']:
            return {reg.name: soqs.pop(reg.name) for reg in bloq.signature.lefts()}

        soqs |= bb.add_d(self.cost_eval_oracle, **_extract_soqs(self.cost_eval_oracle))
        soqs |= bb.add_d(self.phase_oracle, **_extract_soqs(self.phase_oracle))
        cost_eval_adjoint = self.cost_eval_oracle.adjoint()
        soqs |= bb.add_d(cost_eval_adjoint, **_extract_soqs(cost_eval_adjoint))
        return soqs


@attrs.frozen
class PhaseOracleZPow(GateWithRegisters):
    """Phasing oracle that simply applies a ZPow rotation to every qubit in the cost register"""

    cost_reg: Register
    gamma: float = 1.0
    eps: Union[float, sympy.Expr] = 1e-9

    @cached_property
    def cost_dtype(self) -> QFxp:
        dtype = self.cost_reg.dtype
        assert isinstance(dtype, QFxp)
        return dtype

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([self.cost_reg])

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        out = soqs[self.cost_reg.name]
        out = bb.split(out)
        eps = self.eps / len(out)
        if self.cost_dtype.signed:
            out[0] = bb.add(ZPowGate(exponent=1, eps=eps), q=out[0])
        for i in range(self.cost_dtype.bitsize):
            power_of_two = i - self.cost_dtype.num_frac
            out[-(i + 1)] = bb.add(
                ZPowGate(exponent=(2**power_of_two) * self.gamma * 2, eps=self.eps / len(out)),
                q=out[-(i + 1)],
            )
        return {self.cost_reg.name: bb.join(out)}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        zpow = ZPowGate(exponent=self.gamma, eps=self.eps / self.cost_dtype.bitsize)
        return {(zpow, self.cost_dtype.bitsize)}


@attrs.frozen
class PhaseOraclePhaseGradient(GateWithRegisters):
    """Phasing oracle that applies a rotation via addition into the phase gradient register."""

    cost_reg: Register
    gamma: float = 1.0
    eps: Union[float, sympy.Expr] = 1e-9

    def __attrs_post_init__(self):
        dtype = self.cost_reg.dtype
        assert isinstance(dtype, QFxp) and dtype.bitsize == dtype.num_frac

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([self.cost_reg, Register('phase_grad', QFxp(self.b_grad, self.b_grad))])

    @cached_property
    def cost_dtype(self) -> QFxp:
        dtype = self.cost_reg.dtype
        assert isinstance(dtype, QFxp)
        return dtype

    @cached_property
    def b_phase(self) -> int:
        return utils.ceil(utils.log2(1 / self.eps))

    @cached_property
    def b_grad(self) -> int:
        # Using Equation A7 from https://arxiv.org/abs/2007.07391
        return utils.ceil(utils.log2((self.gamma_bitsize + 2) * sympy.pi / self.eps))

    @cached_property
    def gamma_bitsize(self) -> int:
        # Using `gamma_bitsize = log(gamma) + b_{phase} + O(1)` defined b/w equation 34 & 35
        # of https://arxiv.org/abs/2007.07391. Note that the `log(gamma)` here means ignoring
        # leading digits of `gamma` which are 0
        return self.b_phase

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        add_scaled_val = AddScaledValIntoPhaseReg(
            self.cost_dtype.bitsize, self.b_grad, self.gamma, self.gamma_bitsize
        )
        out, phase_grad = bb.add(
            add_scaled_val, x=soqs[self.cost_reg.name], phase_grad=soqs['phase_grad']
        )
        return {self.cost_reg.name: out, 'phase_grad': phase_grad}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        add_scaled_bloq = AddScaledValIntoPhaseReg(
            self.cost_dtype, self.b_grad, self.gamma, self.gamma_bitsize
        )
        return {(add_scaled_bloq, 1)}
