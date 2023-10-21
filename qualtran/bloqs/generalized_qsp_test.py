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
from attrs import frozen
from cirq.testing import random_unitary
from numpy.typing import NDArray

from qualtran import GateWithRegisters, Signature
from qualtran.bloqs.generalized_qsp import GeneralizedQSP, qsp_phase_factors


def test_compute_qsp_phase_factors():
    theta, phi, lambd = qsp_phase_factors([0.5, 0.5], [0.5, -0.5])
    np.testing.assert_allclose(theta, [np.pi / 4, np.pi / 4], atol=1e-7)
    np.testing.assert_allclose(phi, [0, np.pi], atol=1e-7)
    np.testing.assert_allclose(lambd, np.pi, atol=1e-7)


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


@pytest.mark.parametrize("bitsize", [1, 2, 3])
def test_generalized_qsp_on_random_unitaries(bitsize: int):
    random_state = np.random.RandomState(42)

    # TODO generate more example polynomials
    P, Q = [0.5, 0.5], [0.5, -0.5]

    for _ in range(10):
        U = RandomGate.create(bitsize, random_state=random_state)

        result_unitary = cirq.unitary(GeneralizedQSP(U, P, Q))

        expected_top_left = evaluate_polynomial_of_matrix(P, cirq.unitary(U))
        actual_top_left = result_unitary[: 2**bitsize, : 2**bitsize]
        np.testing.assert_allclose(expected_top_left, actual_top_left)

        expected_bottom_left = evaluate_polynomial_of_matrix(Q, cirq.unitary(U))
        actual_bottom_left = result_unitary[2**bitsize :, : 2**bitsize]
        np.testing.assert_allclose(expected_bottom_left, actual_bottom_left)
