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


@pytest.mark.parametrize("degree, precision", [(4, 1e-5), (5, 1e-5)])
def test_complementary_polynomial_quick(degree: int, precision: float):
    random_state = np.random.RandomState(42)
    for _ in range(2):
        P = random_qsp_polynomial(degree, random_state=random_state)
        Q = fast_complementary_polynomial(P, random_state=random_state)
        check_polynomial_pair_on_random_points_on_unit_circle(
            P, Q, random_state=random_state, rtol=precision
        )


@pytest.mark.parametrize("degree, precision", [(3, 1e-2), (4, 1e-1)])
def test_real_polynomial_has_real_complementary_polynomial_quick(degree: int, precision: float):
    random_state = np.random.RandomState(42)

    for _ in range(10):
        P = random_qsp_polynomial(degree, random_state=random_state, only_real_coeffs=True)
        Q = fast_complementary_polynomial(P, random_state=random_state, only_reals=True)
        Q = np.around(Q, decimals=8)
        assert np.isreal(Q).all()
        check_polynomial_pair_on_random_points_on_unit_circle(
            P, Q, random_state=random_state, rtol=precision
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    "degree, num_tests, precision", [(5, 20, 2e-5), (10, 20, 2e-5), (20, 5, 2e-5), (30, 1, 2e-5)]
)
def test_complementary_polynomial(degree: int, num_tests: int, precision: float):
    random_state = np.random.RandomState(42)

    for _ in range(num_tests):
        P = random_qsp_polynomial(degree, random_state=random_state)
        Q = fast_complementary_polynomial(P, random_state=random_state)
        check_polynomial_pair_on_random_points_on_unit_circle(
            P, Q, random_state=random_state, rtol=precision
        )


@pytest.mark.slow
@pytest.mark.parametrize("degree, num_tests", [(2, 10), (3, 10), (4, 10), (5, 10), (10, 10)])
def test_fast_qsp_on_random_unitaries(degree: int, num_tests: int):
    random_state = np.random.RandomState(102)

    for _ in range(num_tests):
        P = random_qsp_polynomial(degree, random_state=random_state)
        U = MatrixGate.random(2, random_state=random_state)
        Q = fast_complementary_polynomial(P, random_state=random_state)
        verify_generalized_qsp(U, P, Q=Q)


@pytest.mark.slow
@pytest.mark.parametrize(
    "degree, num_tests, precision",
    [(2, 10, 2e-2), (3, 10, 2e-2), (4, 10, 5e-2), (5, 10, 2e-2), (10, 10, 2e-2), (20, 10, 2e-2)],
)
def test_real_polynomial_has_real_complementary_polynomial(
    degree: int, num_tests: int, precision: float
):
    random_state = np.random.RandomState(42)
    for _ in range(num_tests):
        P = random_qsp_polynomial(degree, random_state=random_state, only_real_coeffs=True)
        Q = fast_complementary_polynomial(P, random_state=random_state, only_reals=True)
        Q = np.around(Q, decimals=8)
        assert np.isreal(Q).all()
        check_polynomial_pair_on_random_points_on_unit_circle(
            P, Q, random_state=random_state, rtol=precision
        )


@pytest.mark.slow
@pytest.mark.parametrize("bitsize", [1, 2, 3])
@pytest.mark.parametrize(
    "degree, negative_power, tolerance",
    [(2, 0, 1e-5), (2, 1, 1e-5), (2, 2, 1e-5), (5, 0, 1e-4), (5, 1, 1e-4), (5, 2, 1e-4)],
)
def test_generalized_qsp_with_complex_poly_on_random_unitaries(
    bitsize: int, degree: int, negative_power: int, tolerance: float
):
    # TODO Fix high error on degree 20 polynomial
    random_state = np.random.RandomState(42)

    for _ in range(10):
        U = MatrixGate.random(bitsize, random_state=random_state)
        P = random_qsp_polynomial(degree, random_state=random_state)
        Q = fast_complementary_polynomial(P, random_state=random_state)
        verify_generalized_qsp(U, P, negative_power=negative_power, Q=Q, tolerance=tolerance)
