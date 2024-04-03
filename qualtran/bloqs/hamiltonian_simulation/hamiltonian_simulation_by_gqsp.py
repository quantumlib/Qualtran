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
from typing import Dict, Set, Tuple, TYPE_CHECKING

import numpy as np
from attrs import field, frozen
from numpy.typing import NDArray

from qualtran import bloq_example, Controlled, CtrlSpec, GateWithRegisters, Signature
from qualtran.bloqs.basic_gates import SU2RotationGate
from qualtran.bloqs.generalized_qsp import GeneralizedQSP
from qualtran.bloqs.qubitization_walk_operator import QubitizationWalkOperator
from qualtran.linalg.jacobi_anger_approximations import (
    approx_exp_cos_by_jacobi_anger,
    degree_jacobi_anger_approximation,
)
from qualtran.resource_counting.symbolic_counting_utils import (
    is_symbolic,
    SymbolicFloat,
    SymbolicInt,
)

if TYPE_CHECKING:
    from qualtran import BloqBuilder, SoquetT
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


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
        return is_symbolic(self.t, self.alpha, self.precision)

    @cached_property
    def degree(self) -> SymbolicInt:
        r"""degree of the polynomial to approximate the function e^{it\cos(\theta)}"""
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

    def __add_prepare(
        self,
        bb: 'BloqBuilder',
        gqsp_soqs: Dict[str, 'SoquetT'],
        state_prep_ancilla_soqs: Dict[str, 'SoquetT'],
        *,
        adjoint: bool = False,
    ) -> Tuple[Dict[str, 'SoquetT'], Dict[str, 'SoquetT']]:
        prepare = self.walk_operator.prepare

        selection_registers = {reg.name: gqsp_soqs[reg.name] for reg in prepare.selection_registers}
        prepare_out_soqs = bb.add_d(
            prepare.adjoint() if adjoint else prepare,
            **selection_registers,
            **state_prep_ancilla_soqs,
        )
        gqsp_soqs |= {
            reg.name: prepare_out_soqs.pop(reg.name) for reg in prepare.selection_registers
        }
        return gqsp_soqs, prepare_out_soqs

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        # PREPARE, GQSP, PREPARE†
        state_prep_ancilla = {
            reg.name: bb.allocate(reg.total_bits())
            for reg in self.walk_operator.prepare.junk_registers
        }
        soqs, state_prep_ancilla = self.__add_prepare(bb, soqs, state_prep_ancilla)
        soqs = bb.add_d(self.gqsp, **soqs)
        soqs, state_prep_ancilla = self.__add_prepare(bb, soqs, state_prep_ancilla, adjoint=True)

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


@bloq_example
def _hubbard_time_evolution_by_gqsp() -> HamiltonianSimulationByGQSP:
    from qualtran.bloqs.hubbard_model import get_walk_operator_for_hubbard_model

    walk_op = get_walk_operator_for_hubbard_model(2, 2, 1, 1)
    hubbard_time_evolution_by_gqsp = HamiltonianSimulationByGQSP(
        walk_op, t=5, alpha=1, precision=1e-7
    )
    return hubbard_time_evolution_by_gqsp
