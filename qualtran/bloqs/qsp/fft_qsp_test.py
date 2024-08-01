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

import numpy as np
import pytest

from qualtran.bloqs.for_testing.matrix_gate import MatrixGate
from qualtran.bloqs.qsp.fft_qsp import fft_complementary_polynomial
from qualtran.bloqs.qsp.generalized_qsp_test import verify_generalized_qsp
from qualtran.linalg.polynomial.qsp_testing import (
    check_gqsp_polynomial_pair_on_random_points_on_unit_circle,
    random_qsp_polynomial,
)


@pytest.mark.parametrize(
    "degree, num_modes",
    [(3, 500), (4, 500), (5, 500), (10, 500), (40, 500), (100, 500), (1000, 5000), (10000, 50000)],
)
def test_complementary_polynomial(degree: int, num_modes: int):
    random_state = np.random.RandomState(42)
    tolerance = 1e-9
    for _ in range(5):
        P = random_qsp_polynomial(degree, random_state=random_state, only_real_coeffs=False)
        Q = fft_complementary_polynomial(P, num_modes=num_modes, tolerance=tolerance)
        check_gqsp_polynomial_pair_on_random_points_on_unit_circle(
            P, Q, random_state=random_state, rtol=tolerance
        )


@pytest.mark.parametrize(
    "degree, num_modes",
    [(3, 500), (4, 500), (5, 500), (10, 500), (40, 500), (100, 500), (1000, 5000), (10000, 50000)],
)
def test_real_polynomial_has_real_complementary_polynomial(degree: int, num_modes: int):
    random_state = np.random.RandomState(42)
    tolerance = 1e-9
    for _ in range(5):
        P = random_qsp_polynomial(degree, random_state=random_state, only_real_coeffs=True)
        Q = fft_complementary_polynomial(P, tolerance=tolerance, num_modes=num_modes)
        np.testing.assert_allclose(np.imag(Q), 0, atol=tolerance)
        check_gqsp_polynomial_pair_on_random_points_on_unit_circle(
            P, Q, random_state=random_state, rtol=tolerance
        )


@pytest.mark.parametrize("degree", [2, 3, 4, 5, 10])
@pytest.mark.parametrize("bitsize", [1, 2, 3])
def test_fft_qsp_on_random_unitaries(degree: int, bitsize: int):
    random_state = np.random.RandomState(102)
    num_modes = 500
    tolerance = 1e-6
    for _ in range(5):
        P = random_qsp_polynomial(degree, random_state=random_state)
        U = MatrixGate.random(bitsize=bitsize, random_state=random_state)
        Q = fft_complementary_polynomial(P, tolerance=tolerance, num_modes=num_modes)
        verify_generalized_qsp(U, P, Q=Q, tolerance=tolerance)
