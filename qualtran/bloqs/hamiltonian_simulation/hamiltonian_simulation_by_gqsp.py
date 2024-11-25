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
from collections import Counter
from functools import cached_property
from typing import cast, Dict, Tuple, TYPE_CHECKING, Union

import numpy as np
from attrs import field, frozen
from numpy.typing import NDArray

from qualtran import Bloq, bloq_example, BloqDocSpec, CtrlSpec, Signature, Soquet
from qualtran.bloqs.basic_gates.su2_rotation import SU2RotationGate
from qualtran.bloqs.qsp.generalized_qsp import GeneralizedQSP
from qualtran.bloqs.qubitization.qubitization_walk_operator import QubitizationWalkOperator
from qualtran.linalg.polynomial.jacobi_anger_approximations import (
    approx_exp_cos_by_jacobi_anger,
    degree_jacobi_anger_approximation,
)
from qualtran.linalg.polynomial.qsp_testing import scale_down_to_qsp_polynomial
from qualtran.symbolics import is_symbolic, Shaped, SymbolicFloat, SymbolicInt

if TYPE_CHECKING:
    from qualtran import BloqBuilder, SoquetT
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@frozen
class HamiltonianSimulationByGQSP(Bloq):
    r"""Hamiltonian simulation using Generalized QSP given a qubitized quantum walk operator.

    Given the Szegedy Quantum Walk Operator for a Hamiltonian $H$ constructed from SELECT and PREPARE oracles,
    one can construct a block-encoding of $e^{-iHt}$ using GQSP (Corollary 8).

    ### Recap: Qubitization Walk Operator

    For a Hamiltonian $H = \sum_j \alpha_j U_j$ where $U_j$ are unitaries and $\alpha_j \ge 0$,
    we are given the SELECT and PREPARE oracles:
    $$ \text{SELECT} = \sum_j |j\rangle\langle j| \otimes U_j $$
    $$ \text{PREPARE} |0\rangle = \sum_j \frac{\sqrt{\alpha_j}}{\|\alpha\|_1} |j\rangle $$

    We can then implement the [QubitizationWalkOperator](../qubitization_walk_operator.ipynb) that encodes the spectrum of $H$ in the eigenphases of the walk operator $W$.

    ### Approximating the function $e^{i\theta} \mapsto e^{it\cos\theta}$

    We can use the [Jacobi-Anger expansion](https://en.wikipedia.org/wiki/Jacobi%E2%80%93Anger_expansion) to obtain low-degree polynomial approximations for the $\cos$ function:

    $$
        e^{it\cos\theta} = \sum_{n = -\infty}^{\infty} i^n J_n(t) (e^{i\theta})^n
    $$
    where $J_n$ is the $n$-th [Bessel function of the first kind](https://en.wikipedia.org/wiki/Bessel_function#Bessel_functions_of_the_first_kind).

    If we cut-off the above to terms upto degree $d$, we get

    $$
        P[t](z) = \sum_{n = -d}^d i^n J_n(t) z^n
    $$

    Polynomial approximations of the above are provided in the [`qualtran.linalg.jacobi_anger_approximations`](../../linalg/jacobi_anger_approximations.py) module.

    ### Simulation: Block-encoding $e^{-iHt}$

    As the eigenphases of the walk operator above are $e^{-i\arccos(E_k / \|\alpha\|_1)}$,
    we can use the GQSP polynomial with $P = P[-\|\alpha\|_1 t]$ to obtain $P(U) = e^{-iHt}$.
    The obtained GQSP operator $G$ can then be used with two calls to the PREPARE oracle to simulate the hamiltonian:

    $$
        (I \otimes \text{PREPARE}^\dagger \otimes I) G (I \otimes \text{PREPARE} \otimes I) |0\rangle|0\rangle|\psi\rangle = |0\rangle|0\rangle e^{-iHt}|\psi\rangle
    $$

    This therefore block-encodes $e^{-iHt}$ in the block where the signal qubit and selection registers are all $|0\rangle$.

    References:
        [Generalized Quantum Signal Processing](https://arxiv.org/abs/2308.01501)
        Motlagh and Wiebe. (2023). Theorem 7, Corollary 8.

    Args:
        walk_operator: qubitization walk operator of $H$ constructed from SELECT and PREPARE oracles.
        t: time to simulate the Hamiltonian, i.e. $e^{-iHt}$
        precision: the precision $\epsilon$ of the final block encoded $e^{-iHt}$. Split into two:
                   half to approximate $e^{it\cos\theta}$ to a polynomial, and half to synthesize
                   the underlying GQSP rotations.
    """

    walk_operator: QubitizationWalkOperator
    t: SymbolicFloat = field(kw_only=True)
    precision: SymbolicFloat = field(kw_only=True)

    def __attrs_post_init__(self):
        if self.walk_operator.sum_of_lcu_coefficients is None:
            raise ValueError(
                f"Missing attribute `sum_of_ham_coeffs` for {self.walk_operator}, cannot implement Hamiltonian Simulation"
            )

    def is_symbolic(self):
        return is_symbolic(self.t, self.alpha, self.precision)

    @property
    def alpha(self):
        return self.walk_operator.sum_of_lcu_coefficients

    @cached_property
    def degree(self) -> SymbolicInt:
        r"""degree of the polynomial to approximate the function e^{it\cos(\theta)}"""
        return degree_jacobi_anger_approximation(self.t * self.alpha, precision=self.precision / 2)

    @cached_property
    def approx_cos(self) -> Union[NDArray[np.complex128], Shaped]:
        r"""polynomial approximation for $$e^{i\theta} \mapsto e^{it\cos(\theta)}$$"""
        if self.is_symbolic():
            return Shaped((2 * self.degree + 1,))

        poly = approx_exp_cos_by_jacobi_anger(-self.t * self.alpha, degree=cast(int, self.degree))

        # TODO(#860) current scaling method does not compute true maximum, so we scale down a bit more by (1 - 2\eps)
        poly = scale_down_to_qsp_polynomial(list(poly)) * (1 - 2 * self.precision)
        return poly

    @cached_property
    def gqsp(self) -> GeneralizedQSP:
        return GeneralizedQSP.from_qsp_polynomial(
            self.walk_operator,
            self.approx_cos,
            negative_power=self.degree,
            precision=self.precision / 2,
            verify=True,
            verify_precision=1e-4,
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
        state_prep_ancilla: Dict[str, 'SoquetT'] = {
            reg.name: bb.allocate(reg.total_bits())
            for reg in self.walk_operator.prepare.junk_registers
        }

        # PREPARE, GQSP, PREPARE†
        soqs, state_prep_ancilla = self.__add_prepare(bb, soqs, state_prep_ancilla)
        soqs = bb.add_d(self.gqsp, **soqs)
        soqs, state_prep_ancilla = self.__add_prepare(bb, soqs, state_prep_ancilla, adjoint=True)

        for soq in state_prep_ancilla.values():
            if isinstance(soq, Soquet):
                bb.free(soq)
            else:
                for soq_element in soq:
                    bb.free(cast(Soquet, soq_element))

        return soqs

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        counts = Counter[Bloq]()

        d = self.degree
        counts[self.walk_operator.prepare] += 1
        counts[self.walk_operator.prepare.adjoint()] += 1
        counts[self.walk_operator.controlled(ctrl_spec=CtrlSpec(cvs=0))] += d
        counts[self.walk_operator.adjoint().controlled()] += d
        counts[SU2RotationGate.arbitrary(ssa)] += 2 * d + 1

        return counts


@bloq_example
def _hubbard_time_evolution_by_gqsp() -> HamiltonianSimulationByGQSP:
    from qualtran.bloqs.chemistry.hubbard_model.qubitization import (
        get_walk_operator_for_hubbard_model,
    )

    walk_op = get_walk_operator_for_hubbard_model(2, 2, 1, 1)
    hubbard_time_evolution_by_gqsp = HamiltonianSimulationByGQSP(walk_op, t=5, precision=1e-7)
    return hubbard_time_evolution_by_gqsp


@bloq_example
def _symbolic_hamsim_by_gqsp() -> HamiltonianSimulationByGQSP:
    import sympy

    from qualtran.bloqs.chemistry.hubbard_model.qubitization import (
        get_walk_operator_for_hubbard_model,
    )

    tau, t, inv_eps = sympy.symbols(r"\tau t \epsilon^{-1}", positive=True)
    walk_op = get_walk_operator_for_hubbard_model(2, 2, tau, 4 * tau)
    symbolic_hamsim_by_gqsp = HamiltonianSimulationByGQSP(walk_op, t=t, precision=1 / inv_eps)
    return symbolic_hamsim_by_gqsp


_Hamiltonian_Simulation_by_GQSP_DOC = BloqDocSpec(
    bloq_cls=HamiltonianSimulationByGQSP,
    import_line='from qualtran.bloqs.hamiltonian_simulation.hamiltonian_simulation_by_gqsp import HamiltonianSimulationByGQSP',
    examples=[_hubbard_time_evolution_by_gqsp, _symbolic_hamsim_by_gqsp],
)
