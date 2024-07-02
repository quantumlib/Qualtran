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

from .fft_qsp import fft_complementary_polynomial
from .generalized_qsp_test import (
    check_polynomial_pair_on_random_points_on_unit_circle,
    random_qsp_polynomial,
    verify_generalized_qsp,
)


@pytest.mark.parametrize("degree, tolerance", [(4, 1e-3), (5, 1e-3)])
def test_complementary_polynomial_quick(degree: int, tolerance: float):
    random_state = np.random.RandomState(42)
    for _ in range(2):
        P = random_qsp_polynomial(degree, random_state=random_state)
        Q = fft_complementary_polynomial(P, tolerance=tolerance)
        check_polynomial_pair_on_random_points_on_unit_circle(
            P, Q, random_state=random_state, rtol=tolerance
        )


@pytest.mark.parametrize("degree, tolerance", [(3, 1e-3), (4, 1e-3)])
def test_real_polynomial_has_real_complementary_polynomial_quick(degree: int, tolerance: float):
    random_state = np.random.RandomState(42)

    for _ in range(3):
        P = random_qsp_polynomial(degree, random_state=random_state, only_real_coeffs=True)
        Q = fft_complementary_polynomial(P, tolerance)
        Q = np.around(Q, decimals=8)
        assert np.isreal(Q).all()
        check_polynomial_pair_on_random_points_on_unit_circle(
            P, Q, random_state=random_state, rtol=tolerance
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    "degree, num_tests, tolerance", [(5, 3, 2e-3), (10, 1, 2e-2), (20, 1, 2e-2), (30, 1, 1e-1)]
)
def test_complementary_polynomial(degree: int, num_tests: int, tolerance: float):
    random_state = np.random.RandomState(42)

    for _ in range(num_tests):
        P = random_qsp_polynomial(degree, random_state=random_state)
        Q = fft_complementary_polynomial(P, tolerance)
        check_polynomial_pair_on_random_points_on_unit_circle(
            P, Q, random_state=random_state, rtol=tolerance
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    "degree, num_tests, tolerance",
    [(2, 3, 1e-4), (3, 3, 1e-3), (4, 1, 1e-3), (5, 1, 1e-2), (10, 3, 1e-2)],
)
def test_fast_qsp_on_random_unitaries(degree: int, num_tests: int, tolerance: float):
    random_state = np.random.RandomState(102)

    for _ in range(num_tests):
        P = random_qsp_polynomial(degree, random_state=random_state)
        U = MatrixGate.random(2, random_state=random_state)
        Q = fft_complementary_polynomial(P, tolerance=tolerance)
        verify_generalized_qsp(U, P, Q=Q, tolerance=tolerance)


@pytest.mark.slow
@pytest.mark.parametrize(
    "degree, num_tests, tolerance",
    [(2, 3, 2e-4), (3, 3, 2e-3), (4, 3, 1e-3), (5, 2, 1e-2), (10, 2, 1e-2), (20, 1, 2e-2)],
)
def test_real_polynomial_has_real_complementary_polynomial(
    degree: int, num_tests: int, tolerance: float
):
    random_state = np.random.RandomState(42)
    for _ in range(num_tests):
        P = random_qsp_polynomial(degree, random_state=random_state, only_real_coeffs=True)
        Q = fft_complementary_polynomial(P, tolerance=tolerance)
        Q = np.around(Q, decimals=8)
        assert np.isreal(Q).all()
        check_polynomial_pair_on_random_points_on_unit_circle(
            P, Q, random_state=random_state, rtol=tolerance
        )


@pytest.mark.slow
@pytest.mark.parametrize("bitsize", [1, 2, 3])
@pytest.mark.parametrize(
    "degree, negative_power, tolerance",
    [(2, 0, 1e-3), (2, 1, 1e-3), (2, 2, 1e-3), (5, 0, 1e-2), (5, 1, 1e-3), (5, 2, 1e-3)],
)
def test_generalized_qsp_with_complex_poly_on_random_unitaries(
    bitsize: int, degree: int, negative_power: int, tolerance: float
):
    random_state = np.random.RandomState(42)

    for _ in range(3):
        U = MatrixGate.random(bitsize, random_state=random_state)
        P = random_qsp_polynomial(degree, random_state=random_state)
        Q = fft_complementary_polynomial(P, tolerance)
        verify_generalized_qsp(U, P, negative_power=negative_power, Q=Q, tolerance=tolerance)
