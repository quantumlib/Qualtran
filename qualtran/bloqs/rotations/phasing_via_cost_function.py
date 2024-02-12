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
import sympy
from functools import cached_property
from typing import Dict, TYPE_CHECKING, Union

import attrs
import numpy as np

from qualtran import GateWithRegisters, QFxp, Signature, Bloq, BloqBuilder, Register
from qualtran.bloqs.basic_gates.rotation import ZPowGate
from qualtran.bloqs.rotations.phase_gradient import AddScaledValIntoPhaseReg

if TYPE_CHECKING:
    from qualtran import SoquetT


@attrs.frozen
class PhasingViaCostFunction(Bloq):
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


@attrs.frozen
class PhaseOraclePhaseGradient(GateWithRegisters):
    cost_reg: Register
    gamma: float = 1.0
    eps: Union[float, sympy.Expr] = 1e-9

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
        return int(np.ceil(np.log2(1 / self.eps)))

    @cached_property
    def b_grad(self) -> int:
        # Using Equation A7 from https://arxiv.org/abs/2007.07391
        eq_a7 = int(np.ceil(np.log2((self.gamma_bitsize + 2) * np.pi / self.eps)))
        # Using Equation 35 from https://arxiv.org/abs/2007.07391
        eq_35 = self.b_phase + int(np.ceil(np.log2(self.b_phase)))
        assert eq_a7 >= eq_35 >= self.cost_dtype.bitsize
        return eq_35

    @cached_property
    def gamma_bitsize(self) -> int:
        # TODO: Verify that gamma_bitsize computation is correct. The +5 is currently arbitrary to
        #  make tests pass. Paragraph b/w equation 34 & 35 of https://arxiv.org/abs/2007.07391
        #  gives `gamma_bitsize` to be `log(gamma) + b_{phase} + O(1)`
        return self.b_phase + self.cost_dtype.num_frac + 5

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        out, phase_grad = bb.add(
            AddScaledValIntoPhaseReg(self.cost_dtype, self.b_grad, self.gamma, self.gamma_bitsize),
            x=soqs[self.cost_reg.name],
            phase_grad=soqs['phase_grad'],
        )
        return {self.cost_reg.name: out, 'phase_grad': phase_grad}
