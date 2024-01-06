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

    @property
    def signature(self) -> Signature:
        return Signature.build(q=1)

    @property
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

        gates = cirq.single_qubit_matrix_to_gates(self.rotation_matrix)
        matrix = np.eye(2)
        for gate in gates:
            yield gate.on(qubit)
            matrix = cirq.unitary(gate) @ matrix

        # `cirq.single_qubit_matrix_to_gates` does not preserve global phase
        matrix = matrix @ self.rotation_matrix.conj().T
        yield cirq.GlobalPhaseGate(matrix[0, 0].conj()).on()


def qsp_complementary_polynomial(
    P: Sequence[complex], *, verify: bool = False
) -> Sequence[complex]:
    r"""Computes the Q polynomial given P

    Computes polynomial $Q$ of degree at-most that of $P$, satisfying

        $$ \abs{P(e^{i\theta})}^2 + \abs{Q(e^{i\theta})}^2 = 1 $$

    for every $\theta \in \mathbb{R}$.

    The exact method for computing $Q$ is described in the proof of Theorem 4.

    Args:
        P: Co-efficients of a complex polynomial.
        verify: sanity check the computed polynomial roots (defaults to False).

    References:
        [Generalized Quantum Signal Processing](https://arxiv.org/abs/2308.01501)
            Motlagh and Wiebe. (2023). Theorem 4.
    """
    d = len(P) - 1  # degree

    # R(z) = z^d (1 - P^*(z) P(z)) (Eq. 34, 35)
    R = Polynomial.basis(d) - Polynomial(P) * Polynomial(np.conj(P[::-1]))
    roots = R.roots()

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

    c = R.coef[-1]
    scaling_factor = np.sqrt(np.abs(c * np.prod(larger_roots)))

    # pair up roots in `units`
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

    Q = scaling_factor * Polynomial.fromroots(paired_units + smaller_roots)

    return np.around(Q.coef, decimals=10)


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

    for d in reversed(range(n)):
        assert S.shape == (2, d + 1)

        a, b = S[:, d]
        theta[d] = np.arctan2(np.abs(b), np.abs(a))
        phi[d] = np.angle(a) - np.angle(b)

        if d == 0:
            lambd = np.angle(b)
        else:
            S = SU2RotationGate(theta[d], phi[d], 0).rotation_matrix @ S
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

    @property
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
