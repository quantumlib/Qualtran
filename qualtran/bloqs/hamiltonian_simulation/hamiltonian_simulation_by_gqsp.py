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

import numpy as np
import scipy
import sympy
from attrs import field, frozen
from numpy.typing import NDArray

from qualtran import GateWithRegisters, Signature
from qualtran.bloqs.generalized_qsp import GeneralizedQSP
from qualtran.bloqs.qubitization_walk_operator import QubitizationWalkOperator


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
    t: float = field(kw_only=True)
    alpha: float = field(kw_only=True)
    precision: float = field(kw_only=True)

    @cached_property
    def degree(self) -> int:
        r"""degree of the polynomial to approximate the function e^{it\cos(\theta)}"""
        d = 0
        while True:
            term = scipy.special.jv(d + 1, self.t * self.alpha)
            if np.isclose(term, 0, atol=self.precision / 2):
                break
            d += 1
        return d

    @cached_property
    def approx_cos(self) -> NDArray[np.complex_]:
        r"""polynomial approximation for $$e^{i\theta} \mapsto e^{it\cos(\theta)}$$"""
        coeff_indices = np.arange(-self.degree, self.degree + 1)
        approx_cos = 1j**coeff_indices * scipy.special.jv(coeff_indices, self.t * self.alpha)
        return approx_cos

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
        t = ssa.new_symbol('t')
        alpha = ssa.new_symbol('alpha')
        inv_precision = ssa.new_symbol('1/precision')
        d = sympy.O(
            t * alpha + sympy.log(1 / inv_precision) / sympy.log(sympy.log(1 / inv_precision)),
            (t, sympy.oo),
            (alpha, sympy.oo),
            (inv_precision, sympy.oo),
        )

        # TODO account for SU2 rotation gates
        return {(self.walk_operator, d)}
