#  Copyright 2024 Google LLC
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
from typing import Dict, Set

import cirq
import numpy as np
import sympy
from attrs import field, frozen
from numpy.typing import NDArray

from qualtran import Controlled, CtrlSpec, GateWithRegisters, Signature
from qualtran.bloqs.basic_gates import SU2RotationGate
from qualtran.bloqs.generalized_qsp import GeneralizedQSP
from qualtran.bloqs.qsp.polynomial_approximations import (
    approx_exp_cos_by_jacobi_anger,
    degree_jacobi_anger_approximation,
)
from qualtran.bloqs.qubitization_walk_operator import QubitizationWalkOperator
from qualtran.resource_counting.symbolic_counting_utils import SymbolicFloat, SymbolicInt


@frozen
class HamiltonianSimulationByGQSP(GateWithRegisters):
    r"""Hamiltonian simulation using Generalized QSP given a qubitized quantum walk operator.

    Implements Hamiltonian simulation given a walk operator from SELECT and PREPARE oracles.

    We can use the Jacobi-Anger expansion to obtain low-degree polynomial approximations for the $\cos$ function:

        $$ e^{it\cos\theta} = \sum_{n = -\infty}^{\infty} i^n J_n(t) (e^{i\theta})^n $$

    where $J_n$ is the $n$-th [Bessel function of the first kind](https://en.wikipedia.org/wiki/Bessel_function#Bessel_functions_of_the_first_kind), which is provided by `scipy.special.jv`.
    We can cutoff at $d = O(t + \log(1/\epsilon) / \log\log(1/\epsilon))$ to get an $\epsilon$-approximation (Theorem 7):

        $$ P[t](z) = \sum_{n = -d}^d i^n J_n(t) z^n $$

    References:
        [Generalized Quantum Signal Processing](https://arxiv.org/abs/2308.01501)
            Motlagh and Wiebe. (2023). Theorem 7.
    """

    walk_operator: QubitizationWalkOperator
    t: SymbolicFloat = field(kw_only=True)
    alpha: SymbolicFloat = field(kw_only=True)
    precision: SymbolicFloat = field(kw_only=True)

    def _parameterized_(self):
        return cirq.is_parameterized((self.t, self.alpha, self.precision))

    @cached_property
    def degree(self) -> SymbolicInt:
        r"""degree of the polynomial to approximate the function e^{it\cos(\theta)}"""
        if self._parameterized_():
            return sympy.O(
                self.t * self.alpha
                + sympy.log(self.precision) / sympy.log(sympy.log(self.precision))
            )
        else:
            return degree_jacobi_anger_approximation(self.t * self.alpha, precision=self.precision)

    @cached_property
    def approx_cos(self) -> NDArray[np.complex_]:
        r"""polynomial approximation for $$e^{i\theta} \mapsto e^{it\cos(\theta)}$$"""
        if self._parameterized_():
            raise ValueError(f"cannot compute `cos` approximation for parameterized Bloq {self}")
        return approx_exp_cos_by_jacobi_anger(self.t * self.alpha, degree=self.degree)

    @cached_property
    def gqsp(self) -> GeneralizedQSP:
        return GeneralizedQSP.from_qsp_polynomial(
            self.walk_operator, self.approx_cos, negative_power=self.degree
        )

    @cached_property
    def signature(self) -> 'Signature':
        return self.gqsp.signature

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        prepare = self.walk_operator.prepare
        state_prep_ancilla = {
            reg.name: bb.allocate(reg.total_bits()) for reg in prepare.junk_registers
        }

        # PREPARE
        prepare_soqs = bb.add_d(
            self.walk_operator.prepare, selection=soqs['selection'], **state_prep_ancilla
        )
        soqs['selection'] = prepare_soqs.pop('selection')
        state_prep_ancilla = prepare_soqs

        # GQSP
        soqs = bb.add_d(self.gqsp, **soqs)

        # PREPARE†
        prepare_soqs = bb.add_d(
            self.walk_operator.prepare.adjoint(), selection=soqs['selection'], **state_prep_ancilla
        )
        soqs['selection'] = prepare_soqs.pop('selection')
        state_prep_ancilla = prepare_soqs

        for soq in state_prep_ancilla.values():
            bb.free(soq)

        return soqs

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        if self._parameterized_():
            d = self.degree
            return {
                (Controlled(self.walk_operator.adjoint(), CtrlSpec()), d),
                (Controlled(self.walk_operator, CtrlSpec(cvs=0)), d),
                (self.walk_operator.prepare, 1),
                (self.walk_operator.prepare.adjoint(), 1),
                (SU2RotationGate.arbitrary(ssa), 2 * d + 1),
            }
        return self.decompose_bloq().build_call_graph(ssa)
