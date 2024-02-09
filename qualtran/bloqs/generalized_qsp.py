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
from typing import Sequence, Tuple

import cirq
import numpy as np
from attrs import frozen
from numpy.polynomial import Polynomial
from numpy.typing import NDArray

from qualtran import GateWithRegisters, Register, Signature


@frozen
class SU2RotationGate(GateWithRegisters):
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

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, q: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        qubit = q[0]

        yield cirq.Rz(rads=np.pi - self.lambd).on(qubit)
        yield cirq.Ry(rads=2 * self.theta).on(qubit)
        yield cirq.Rz(rads=-self.phi).on(qubit)
        yield cirq.GlobalPhaseGate(np.exp(1j * (np.pi + self.lambd + self.phi) / 2)).on()


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
        if np.allclose(np.abs(r), 1):
            units.append(r)
        elif np.abs(r) > 1:
            larger_roots.append(r)
        else:
            smaller_roots.append(r)

    if verify:
        # verify that the non-unit roots indeed occur in conjugate pairs.
        def is_permutation(A, B):
            assert len(A) == len(B)
            A = list(A)
            for z in B:
                for w in A:
                    if np.allclose(z, w):
                        A.remove(w)
                        break
                else:
                    return False
            return True

        assert is_permutation(smaller_roots, 1 / np.array(larger_roots).conj())

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

        if matched_z:
            paired_units.append(z)
            unpaired_units.remove(matched_z)
        else:
            unpaired_units.append(z)

    if verify:
        assert len(unpaired_units) == 0

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
        return 0 if np.isclose(x, 0) else np.angle(x)

    for d in reversed(range(n)):
        assert S.shape == (2, d + 1)

        a, b = S[:, d]
        theta[d] = np.arctan2(np.abs(b), np.abs(a))
        # \phi_d = arg(a / b)
        phi[d] = 0 if np.isclose(np.abs(b), 0) else safe_angle(a * np.conj(b))

        if d == 0:
            lambd = safe_angle(b)
        else:
            S = SU2RotationGate(theta[d], phi[d], 0).rotation_matrix.conj().T @ S
            S = np.array([S[0][1 : d + 1], S[1][0:d]])

    return theta, phi, lambd


@frozen
class GeneralizedQSP(GateWithRegisters):
    r"""Applies a QSP polynomial $P$ to a unitary $U$ to obtain a block-encoding of $P(U)$.

    This gate represents the following unitary:

        $$ \begin{bmatrix} P(U) & \cdot \\ \cdot & \cdot \end{bmatrix} $$

    The polynomial $P$ must satisfy:
    $\abs{P(e^{i \theta})}^2 \le 1$ for every $\theta \in \mathbb{R}$.

    The exact circuit is described in Figure 2.

    Args:
        U: Unitary operation.
        P: Co-efficients of a complex polynomial.

    References:
        [Generalized Quantum Signal Processing](https://arxiv.org/abs/2308.01501)
            Motlagh and Wiebe. (2023). Theorem 3; Figure 2.
    """

    U: GateWithRegisters
    P: Sequence[complex]

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register(name='signal', bitsize=1), *self.U.signature])

    @cached_property
    def Q(self):
        return qsp_complementary_polynomial(self.P)

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

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, signal, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        assert len(signal) == 1
        signal_qubit = signal[0]

        yield SU2RotationGate(self._theta[0], self._phi[0], self._lambda).on(signal_qubit)
        for theta, phi in zip(self._theta[1:], self._phi[1:]):
            yield self.U.on_registers(**quregs).controlled_by(signal_qubit, control_values=[0])
            yield SU2RotationGate(theta, phi, 0).on(signal_qubit)
