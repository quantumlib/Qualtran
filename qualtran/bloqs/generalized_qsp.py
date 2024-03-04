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
from typing import Dict, Sequence, Set, Tuple, TYPE_CHECKING

import cirq
import numpy as np
import scipy
import sympy
from attrs import field, frozen
from numpy.polynomial import Polynomial
from numpy.typing import NDArray

from qualtran import Bloq, CtrlSpec, GateWithRegisters, QBit, Register, Signature
from qualtran.bloqs.basic_gates import Ry, ZPowGate
from qualtran.bloqs.qubitization_walk_operator import QubitizationWalkOperator

if TYPE_CHECKING:
    from qualtran import BloqBuilder, SoquetT
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@frozen
class SU2RotationGate(Bloq):
    theta: float
    phi: float
    lambd: float

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(q=1)

    @cached_property
    def rotation_matrix(self):
        r"""Implements an arbitrary SU(2) rotation.

        The rotation is represented by the matrix:

            $$
            \begin{matrix}
            e^{i(\lambda + \phi)} \cos(\theta) & e^{i\phi} \sin(\theta) \\
            e^{i\lambda} \sin(\theta) & - \cos(\theta)
            \end{matrix}
            $$

        Returns:
            A 2x2 rotation matrix

        References:
            [Generalized Quantum Signal Processing](https://arxiv.org/abs/2308.01501)
                Motlagh and Wiebe. (2023). Equation 7.
        """
        return np.array(
            [
                [
                    np.exp(1j * (self.lambd + self.phi)) * np.cos(self.theta),
                    np.exp(1j * self.phi) * np.sin(self.theta),
                ],
                [np.exp(1j * self.lambd) * np.sin(self.theta), -np.cos(self.theta)],
            ]
        )

    def build_composite_bloq(self, bb: 'BloqBuilder', q: 'SoquetT') -> Dict[str, 'SoquetT']:
        q = bb.add(ZPowGate(exponent=2, global_shift=0.5), q=q)
        q = bb.add(ZPowGate(exponent=1 - self.lambd / np.pi, global_shift=-1), q=q)
        q = bb.add(Ry(angle=2 * self.theta), q=q)
        q = bb.add(ZPowGate(exponent=-self.phi / np.pi, global_shift=-1), q=q)
        return {'q': q}


def qsp_complementary_polynomial(
    P: Sequence[complex], *, verify: bool = False
) -> Sequence[complex]:
    r"""Computes the Q polynomial given P

    Computes polynomial $Q$ of degree at-most that of $P$, satisfying

        $$ \abs{P(e^{i\theta})}^2 + \abs{Q(e^{i\theta})}^2 = 1 $$

    for every $\theta \in \mathbb{R}$.

    The exact method for computing $Q$ is described in the proof of Theorem 4.
    The method computes an auxillary polynomial R, whose roots are computed
    and re-interpolated to obtain the required polynomial Q.

    TODO: Also implement the more efficient optimization-based method described in Eq. 52

    Args:
        P: Co-efficients of a complex polynomial.
        verify: sanity check the computed polynomial roots (defaults to False).

    References:
        [Generalized Quantum Signal Processing](https://arxiv.org/abs/2308.01501)
            Motlagh and Wiebe. (2023). Theorem 4.
    """
    d = len(P) - 1  # degree

    # R(z) = z^d (1 - P^*(z) P(z))
    # obtained on simplifying Eq. 34, Eq. 35 by substituting H, T.
    # The definition of $R$ is given in the text from Eq.34 - Eq. 35,
    # following the chain of definitions below:
    #
    #     $$
    #     T(\theta) = \abs{P(e^{i\theta}),
    #     H = I - T,
    #     H = e^{-id\theta} R(e^{i\theta})
    #     $$
    #
    # Here H and T are defined on reals, so the initial definition of R is only on the unit circle.
    # We analytically continue this definition to the entire complex plane by replacing $e^{i\theta}$ by $z$.
    R = Polynomial.basis(d) - Polynomial(P) * Polynomial(np.conj(P[::-1]))
    roots = R.roots()

    # R is self-inversive, so larger_roots and smaller_roots occur in conjugate pairs.
    units: list[complex] = []  # roots r s.t. \abs{r} = 1
    larger_roots: list[complex] = []  # roots r s.t. \abs{r} > 1
    smaller_roots: list[complex] = []  # roots r s.t. \abs{r} < 1

    for r in roots:
        assert not (np.abs(r) <= 1e-5), "zero root!"
        if np.allclose(np.abs(r), 1):
            units.append(r)
        elif np.abs(r) > 1:
            larger_roots.append(r)
        else:
            smaller_roots.append(r)

    # pair up roots in `units`, claimed in Eq. 40 and the explanation preceding it.
    # all unit roots must have even multiplicity.
    paired_units: list[complex] = []
    unpaired_units: list[complex] = []
    for z in units:
        matched_z = None
        for w in unpaired_units:
            if np.allclose(z, w):
                matched_z = w
                break

        if matched_z is not None:
            paired_units.append(z)
            unpaired_units.remove(matched_z)
        else:
            unpaired_units.append(z)

    unpaired_conj_units: list[complex] = []
    for z in unpaired_units:
        matched_z_conj = None
        for w in unpaired_conj_units:
            if np.allclose(z.conjugate(), w):
                matched_z_conj = w
                break

        if matched_z_conj is not None:
            smaller_roots.append(z)
            larger_roots.append(matched_z_conj)
            unpaired_conj_units.remove(matched_z_conj)
        else:
            unpaired_conj_units.append(z)

    if verify:
        assert len(unpaired_conj_units) == 0

        # verify that the non-unit roots indeed occur in conjugate pairs.
        def assert_is_permutation(A, B):
            assert len(A) == len(B)
            A = list(A)
            unmatched = []
            for z in B:
                for w in A:
                    if np.allclose(z, w, rtol=1e-5, atol=1e-5):
                        A.remove(w)
                        break
                else:
                    unmatched.append(z)
            assert len(unmatched) == 0

        assert_is_permutation(smaller_roots, 1 / np.array(larger_roots).conj())

    # Q = G \hat{G}, where
    # - \hat{G}^2 is the monomials which are unit roots of R, which occur in pairs.
    # - G*(z) G(z) is the interpolation of the conjugate paired non-unit roots of R,
    #   described in Eq. 37 - Eq. 38

    # Leading co-efficient of R described in Eq. 37.
    # Note: In the paper, the polynomial is interpolated from larger_roots,
    #       but this is swapped in our implementation to reduce the error in Q.
    c = R.coef[-1]
    scaling_factor = np.sqrt(np.abs(c * np.prod(larger_roots)))

    Q = scaling_factor * Polynomial.fromroots(paired_units + smaller_roots)

    return Q.coef


def qsp_phase_factors(
    P: Sequence[complex], Q: Sequence[complex]
) -> Tuple[Sequence[float], Sequence[float], float]:
    """Computes the QSP signal rotations for a given pair of polynomials.

    The QSP transformation is described in Theorem 3, and the algorithm for computing
    co-efficients is described in Algorithm 1.

    Args:
        P: Co-efficients of a complex polynomial.
        Q: Co-efficients of a complex polynomial.

    Returns:
        A tuple (theta, phi, lambda).
        theta and phi have length degree(P) + 1.

    Raises:
        ValueError: when P and Q have different degrees.

    References:
        [Generalized Quantum Signal Processing](https://arxiv.org/abs/2308.01501)
            Motlagh and Wiebe. (2023). Theorem 3; Algorithm 1.
    """
    if len(P) != len(Q):
        raise ValueError("Polynomials P and Q must have the same degree.")

    S = np.array([P, Q])
    n = S.shape[1]

    theta = np.zeros(n)
    phi = np.zeros(n)
    lambd = 0

    def safe_angle(x):
        return 0 if np.isclose(x, 0, atol=1e-10) else np.angle(x)

    for d in reversed(range(n)):
        assert S.shape == (2, d + 1)

        a, b = S[:, d]
        theta[d] = np.arctan2(np.abs(b), np.abs(a))
        # \phi_d = arg(a / b)
        phi[d] = 0 if np.isclose(np.abs(b), 0, atol=1e-10) else safe_angle(a * np.conj(b))

        if d == 0:
            lambd = safe_angle(b)
        else:
            S = SU2RotationGate(theta[d], phi[d], 0).rotation_matrix.conj().T @ S
            S = np.array([S[0][1 : d + 1], S[1][0:d]])

    return theta, phi, lambd


@frozen
class GeneralizedQSP(GateWithRegisters):
    r"""Applies a QSP polynomial $P$ to a unitary $U$ to obtain a block-encoding of $P(U)$.

    Can optionally provide a negative power offset $k$ (defaults to 0),
    to obtain $U^{-k} P(U)$. (Theorem 6)
    This gate represents the following unitary:

        $$ \begin{bmatrix} U^{-k} P(U) & \cdot \\ Q(U) & \cdot \end{bmatrix} $$

    The polynomial $P$ must satisfy:
    $\abs{P(e^{i \theta})}^2 \le 1$ for every $\theta \in \mathbb{R}$.

    The exact circuit is described in Figure 2.

    Args:
        U: Unitary operation.
        P: Co-efficients of a complex polynomial.
        negative_power: value of $k$, which effectively applies $z^{-k} P(z)$. defaults to 0.

    References:
        [Generalized Quantum Signal Processing](https://arxiv.org/abs/2308.01501)
            Motlagh and Wiebe. (2023). Theorem 3; Figure 2; Theorem 6.
    """

    U: GateWithRegisters
    P: Sequence[complex]
    Q: Sequence[complex]
    negative_power: int = field(default=0, kw_only=True)

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('signal', QBit()), *self.U.signature])

    @classmethod
    def from_qsp_polynomial(
        cls, U: GateWithRegisters, P: Sequence[complex], *, negative_power: int = 0
    ) -> 'GeneralizedQSP':
        Q = qsp_complementary_polynomial(P)
        return GeneralizedQSP(U, P, Q, negative_power=negative_power)

    @cached_property
    def _qsp_phases(self) -> Tuple[Sequence[float], Sequence[float], float]:
        return qsp_phase_factors(self.P, self.Q)

    @cached_property
    def _theta(self) -> Sequence[float]:
        return self._qsp_phases[0]

    @cached_property
    def _phi(self) -> Sequence[float]:
        return self._qsp_phases[1]

    @cached_property
    def _lambda(self) -> float:
        return self._qsp_phases[2]

    @cached_property
    def signal_rotations(self) -> NDArray[SU2RotationGate]:
        return np.array(
            [
                SU2RotationGate(theta, phi, self._lambda if i == 0 else 0)
                for i, (theta, phi) in enumerate(zip(self._theta, self._phi))
            ]
        )

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, signal, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        signal_qubit = signal[0]

        num_inverse_applications = self.negative_power

        yield self.signal_rotations[0].on(signal_qubit)
        for signal_rotation in self.signal_rotations[1:]:
            if num_inverse_applications > 0:
                # apply C-U^\dagger
                yield self.U.adjoint().on_registers(**quregs).controlled_by(signal_qubit)
                num_inverse_applications -= 1
            else:
                # apply C[0]-U
                yield self.U.on_registers(**quregs).controlled_by(signal_qubit, control_values=[0])
            yield signal_rotation.on(signal_qubit)

        for _ in range(num_inverse_applications):
            yield self.U.adjoint().on_registers(**quregs)

    def __hash__(self):
        return hash((self.U, *self.signal_rotations, self.negative_power))

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        degree = len(self.P)
        return {
            (self.U.adjoint().controlled(), min(degree, self.negative_power)),
            (self.U.controlled(ctrl_spec=CtrlSpec(cvs=0)), max(0, degree - self.negative_power)),
            (self.U.adjoint(), max(0, self.negative_power - degree)),
        } | {(rotation, 1) for rotation in self.signal_rotations}


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
        return len(self.approx_cos) // 2

    @cached_property
    def approx_cos(self) -> NDArray[np.complex_]:
        r"""polynomial approximation for $$e^{i\theta} \mapsto e^{it\cos(\theta)}$$"""
        d = 0
        while True:
            term = scipy.special.jv(d + 1, self.t * self.alpha)
            if np.isclose(term, 0, atol=self.precision / 2):
                break
            d += 1

        coeff_indices = np.arange(-d, d + 1)
        approx_cos = 1j**coeff_indices * scipy.special.jv(coeff_indices, self.t * self.alpha)
        return approx_cos

    @cached_property
    def gqsp(self) -> GeneralizedQSP:
        # return GeneralizedQSP.from_qsp_polynomial(
        #     self.walk_operator, self.approx_cos / np.sqrt(2), negative_power=self.degree
        # )
        return GeneralizedQSP(
            self.walk_operator,
            self.approx_cos / np.sqrt(2),
            self.approx_cos / np.sqrt(2),
            negative_power=self.degree,
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
