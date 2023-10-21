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
from numpy.typing import NDArray

from qualtran import GateWithRegisters, Register, Signature


def _arbitrary_SU2_rotation(theta: float, phi: float, lambd: float):
    r"""Implements an arbitrary SU(2) rotation.

    The rotation is represented by the matrix:

        $$
        \begin{matrix}
        e^{i(\lambda + \phi)} \cos(\theta) & e^{i\phi} \sin(\theta) \\
        e^{i\phi} \sin(\theta) & - \cos(\theta)
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
            [np.exp(1j * (lambd + phi)) * np.cos(theta), np.exp(1j * phi) * np.sin(theta)],
            [np.exp(1j * lambd) * np.sin(theta), -np.cos(theta)],
        ]
    )


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
        theta[d] = np.arctan2(np.abs(a), np.abs(b))
        phi[d] = np.angle(a / b)

        if d == 0:
            lambd = np.angle(b)
        else:
            S = _arbitrary_SU2_rotation(theta[d], phi[d], 0) @ S
            S = np.array([S[0][1 : d + 1], S[1][0:d]])

    return theta, phi, lambd


@frozen
class GeneralizedQSP(GateWithRegisters):
    r"""Applies a QSP polynomial $P$ to a unitary $U$ to obtain a block-encoding of $P(U)$.

    Given a pair of QSP polynomials $P, Q$, this gate represents the following unitary:

        $$ \begin{bmatrix} P(U) & \cdot \\ Q(U) & \cdot \end{bmatrix} $$

    The polynomials $P, Q$ must satisfy:
        $\abs{P(x)}^2 + \abs{Q(x)}^2 = 1$ for every $x \in \mathbb{C}$ such that $\abs{x} = 1$

    The exact circuit is described in Figure 2.

    Args:
        U: Unitary operation.
        P: Co-efficients of a complex polynomial.
        Q: Co-efficients of a complex polynomial.

    References:
        [Generalized Quantum Signal Processing](https://arxiv.org/abs/2308.01501)
            Motlagh and Wiebe. (2023). Theorem 3; Figure 2.
    """

    U: GateWithRegisters
    P: Sequence[complex]
    Q: Sequence[complex]

    @property
    def signature(self) -> Signature:
        return Signature([Register(name='signal', bitsize=1), *self.U.signature])

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

    def _signal_gate(
        self, signal_qubit: cirq.Qid, theta: float, phi: float, lambd: float
    ) -> cirq.OP_TREE:
        gates = cirq.single_qubit_matrix_to_gates(_arbitrary_SU2_rotation(theta, phi, lambd))
        for gate in gates:
            yield gate.on(signal_qubit)

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, signal, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        assert len(signal) == 1
        signal_qubit = signal[0]

        yield from self._signal_gate(signal_qubit, self._theta[0], self._phi[0], self._lambda)
        for theta, phi in zip(self._theta[1:], self._phi[1:]):
            yield self.U.on_registers(**quregs).controlled_by(signal_qubit, control_values=[0])
            yield from self._signal_gate(signal_qubit, theta, phi, 0)
