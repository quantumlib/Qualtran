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

from .fast_qsp import fast_complementary_polynomial
from .generalized_qsp_test import (
    check_polynomial_pair_on_random_points_on_unit_circle,
    random_qsp_polynomial,
    verify_generalized_qsp,
)


@pytest.mark.parametrize("degree", [4, 5])
@pytest.mark.parametrize("precision", [1e-5, 1e-5])
def test_complementary_polynomial_quick(degree: int, precision: float):
    random_state = np.random.RandomState(42)
    for _ in range(2):
        P = random_qsp_polynomial(degree, random_state=random_state)
        Q = fast_complementary_polynomial(P)
        check_polynomial_pair_on_random_points_on_unit_circle(
            P, Q, random_state=random_state, rtol=precision
        )


@pytest.mark.parametrize("degree", [3, 4])
@pytest.mark.parametrize("precision", [1e-4, 1e-4])
def test_real_polynomial_has_real_complementary_polynomial_quick(degree: int, precision: float):
    random_state = np.random.RandomState(42)

    for _ in range(10):
        P = random_qsp_polynomial(degree, random_state=random_state, only_real_coeffs=True)
        Q = fast_complementary_polynomial(P, only_reals=True)
        Q = np.around(Q, decimals=8)
        assert np.isreal(Q).all()
        check_polynomial_pair_on_random_points_on_unit_circle(
            P, Q, random_state=random_state, rtol=precision
        )


@pytest.mark.slow
@pytest.mark.parametrize("degree", [5, 10, 20, 30])
@pytest.mark.parametrize("num_tests", [20, 20, 5, 3])
@pytest.mark.parametrize("precision", [1e-5, 1e-5, 1e-5, 1e-5])
def test_complementary_polynomial(degree: int, num_tests: int, precision: float):
    random_state = np.random.RandomState(42)

    results = []
    for _ in range(num_tests):
        P = random_qsp_polynomial(degree, random_state=random_state)
        Q = fast_complementary_polynomial(P)
        results.append(np.array(Q))
        check_polynomial_pair_on_random_points_on_unit_circle(
            P, Q, random_state=random_state, rtol=precision
        )


@pytest.mark.slow
@pytest.mark.parametrize("degree", [2, 3, 4, 5, 10])
@pytest.mark.parametrize("num_tests", [10, 10, 10, 10, 10])
def test_generalized_real_qsp_with_symbolic_signal_matrix(degree: int, num_tests: int):
    random_state = np.random.RandomState(102)

    for _ in range(num_tests):
        P = random_qsp_polynomial(degree, random_state=random_state)
        U = MatrixGate.random(2, random_state=random_state)
        Q = fast_complementary_polynomial(P)
        verify_generalized_qsp(U, P, Q=Q)


@pytest.mark.slow
@pytest.mark.parametrize("degree", [2, 3, 4, 5, 10, 20])
@pytest.mark.parametrize("num_tests", [10, 10, 10, 10, 10, 10])
@pytest.mark.parametrize("precision", [2e-2, 2e-2, 2e-2, 2e-2, 2e-2, 2e-2])
def test_real_polynomial_has_real_complementary_polynomial(
    degree: int, num_tests: int, precision: float
):
    random_state = np.random.RandomState(42)
    for _ in range(num_tests):
        P = random_qsp_polynomial(degree, random_state=random_state, only_real_coeffs=True)
        Q = fast_complementary_polynomial(P, only_reals=True)
        Q = np.around(Q, decimals=8)
        assert np.isreal(Q).all()
        check_polynomial_pair_on_random_points_on_unit_circle(
            P, Q, random_state=random_state, rtol=precision
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    "bitsize, degree, negative_power, tolerance",
    [
        (1, 2, 0, 1e-5),
        (1, 2, 1, 1e-5),
        (1, 2, 2, 1e-5),
        (2, 2, 0, 1e-5),
        (2, 2, 1, 1e-5),
        (2, 2, 2, 1e-5),
        (1, 5, 2, 1e-4),
        (1, 5, 0, 1e-4),
        (3, 5, 0, 1e-4),
        (3, 5, 2, 1e-4),
        (2, 20, 1, 1e-1),
    ],
)
def test_generalized_qsp_with_complex_poly_on_random_unitaries(
    bitsize: int, degree: int, negative_power: int, tolerance: float
):
    random_state = np.random.RandomState(42)

    for _ in range(10):
        U = MatrixGate.random(bitsize, random_state=random_state)
        P = random_qsp_polynomial(degree, random_state=random_state)
        Q = fast_complementary_polynomial(P)
        verify_generalized_qsp(U, P, negative_power=negative_power, Q=Q, tolerance=tolerance)
