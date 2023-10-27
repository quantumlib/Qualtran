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
from typing import Sequence

import cirq
import numpy as np
import pytest
import sympy
from attrs import frozen
from cirq.testing import random_unitary
from numpy.typing import NDArray

from qualtran import GateWithRegisters, Signature
from qualtran.bloqs.generalized_qsp import GeneralizedQSP, qsp_phase_factors, SU2RotationGate


def test_cirq_decompose_SU2_to_single_qubit_pauli_gates():
    rng = np.random.default_rng(42)

    for _ in range(20):
        theta = rng.random() * 2 * np.pi
        phi = rng.random() * 2 * np.pi
        lambd = rng.random() * 2 * np.pi

        gate = SU2RotationGate(theta, phi, lambd)

        expected = gate.rotation_matrix
        actual = cirq.unitary(gate)
        np.testing.assert_allclose(actual, expected)


@frozen
class RandomGate(GateWithRegisters):
    bitsize: int
    matrix: NDArray

    @staticmethod
    def create(bitsize: int, *, random_state=None) -> 'RandomGate':
        matrix = random_unitary(2**bitsize, random_state=random_state)
        return RandomGate(bitsize, matrix)

    @property
    def signature(self) -> Signature:
        return Signature.build(q=self.bitsize)

    def _unitary_(self):
        return self.matrix


def evaluate_polynomial_of_matrix(P: Sequence[complex], U: NDArray) -> NDArray:
    assert U.ndim == 2 and U.shape[0] == U.shape[1]

    pow_U = np.identity(U.shape[0], dtype=U.dtype)
    result = np.zeros(U.shape, dtype=U.dtype)

    for c in P:
        result += pow_U * c
        pow_U = pow_U @ U

    return result


def verify_generalized_qsp(U: GateWithRegisters, P: Sequence[complex], Q: Sequence[complex]):
    input_unitary = cirq.unitary(U)
    N = input_unitary.shape[0]
    result_unitary = cirq.unitary(GeneralizedQSP(U, P, Q))

    expected_top_left = evaluate_polynomial_of_matrix(P, input_unitary)
    actual_top_left = result_unitary[:N, :N]
    np.testing.assert_allclose(expected_top_left, actual_top_left)

    expected_bottom_left = evaluate_polynomial_of_matrix(Q, input_unitary)
    actual_bottom_left = result_unitary[N:, :N]
    np.testing.assert_allclose(expected_bottom_left, actual_bottom_left)


@pytest.mark.parametrize("bitsize", [1, 2, 3])
@pytest.mark.parametrize(
    ("P", "Q"),
    [
        ([0.5, 0.5], [-0.5, 0.5]),
        ([0.5, 0, 0.5], [-0.5, 0, 0.5]),
        ([0.5, 0, 0, 0.5], [-0.5, 0, 0, 0.5]),
    ],
)
def test_generalized_qsp_on_random_unitaries(
    bitsize: int, P: Sequence[complex], Q: Sequence[complex]
):
    random_state = np.random.RandomState(42)

    for _ in range(20):
        U = RandomGate.create(bitsize, random_state=random_state)
        verify_generalized_qsp(U, P, Q)


@frozen
class SymbolicGQSP:
    r"""Run the Generalized QSP algorithm on a symbolic input unitary

    Assuming the input unitary $U$ is symbolic, we construct the signal matrix $A$ as:

        $$
        A = \begin{bmatrix} U & 0 \\ 0 & 1 \end{bmatrix}
        $$

    This matrix $A$ is used to symbolically compute the GQSP transformed unitary, and the
    norms of the output blocks are upper-bounded. We then check if this upper-bound is negligible.
    """

    P: Sequence[complex]
    Q: Sequence[complex]

    def get_symbolic_qsp_matrix(self, U: sympy.Symbol):
        theta, phi, lambd = qsp_phase_factors(self.P, self.Q)

        # 0-controlled U gate
        A = np.array([[U, 0], [0, 1]])

        # compute the final GQSP unitary
        # mirrors the implementation of GeneralizedQSP::decompose_from_registers,
        # as cirq.unitary does not support symbolic matrices
        W = SU2RotationGate(theta[0], phi[0], lambd).rotation_matrix
        for t, p in zip(theta[1:], phi[1:]):
            W = A @ W
            W = SU2RotationGate(t, p, 0).rotation_matrix @ W

        return W

    @staticmethod
    def upperbound_matrix_norm(M, U) -> float:
        return float(sum(abs(c) for c in sympy.Poly(M, U).coeffs()))

    def verify(self):
        U = sympy.symbols('U')
        W = self.get_symbolic_qsp_matrix(U)
        actual_PU, actual_QU = W[:, 0]

        expected_PU = sympy.Poly(reversed(self.P), U)
        expected_QU = sympy.Poly(reversed(self.Q), U)

        error_PU = self.upperbound_matrix_norm(expected_PU - actual_PU, U)
        error_QU = self.upperbound_matrix_norm(expected_QU - actual_QU, U)

        assert abs(error_PU) <= 1e-10
        assert abs(error_QU) <= 1e-10


@pytest.mark.parametrize(
    ("P", "Q"),
    [
        ([0.5, 0.5], [-0.5, 0.5]),
        ([0.5, 0, 0.5], [-0.5, 0, 0.5]),
        ([0.5, 0, 0, 0.5], [-0.5, 0, 0, 0.5]),
    ],
)
def test_generalized_qsp_with_symbolic_signal_matrix(P: Sequence[complex], Q: Sequence[complex]):
    SymbolicGQSP(P, Q).verify()
