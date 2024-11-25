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
from qualtran.bloqs.qsp.fast_qsp import fast_complementary_polynomial
from qualtran.bloqs.qsp.generalized_qsp_test import verify_generalized_qsp
from qualtran.linalg.polynomial.qsp_testing import (
    check_gqsp_polynomial_pair_on_random_points_on_unit_circle,
    random_qsp_polynomial,
)


@pytest.mark.parametrize(
    "degree, num_tests",
    [
        pytest.param(
            degree, 1 if degree >= 20 else 5, marks=pytest.mark.slow if degree >= 20 else ()
        )
        for degree in [4, 5, 8, 20, 30]
    ],
)
def test_complementary_polynomial(degree: int, num_tests: int):
    precision = 1e-5
    random_state = np.random.RandomState(42)

    for _ in range(num_tests):
        P = random_qsp_polynomial(degree, random_state=random_state)
        Q = fast_complementary_polynomial(P, random_state=random_state)
        check_gqsp_polynomial_pair_on_random_points_on_unit_circle(
            P, Q, random_state=random_state, rtol=precision
        )


@pytest.mark.parametrize("degree", [2, 3, 4, 5, 10])
def test_fast_qsp_on_random_unitaries(degree: int):
    random_state = np.random.RandomState(102)

    for _ in range(5):
        P = random_qsp_polynomial(degree, random_state=random_state)
        U = MatrixGate.random(2, random_state=random_state)
        Q = fast_complementary_polynomial(P, random_state=random_state)
        verify_generalized_qsp(U, P, Q=Q)


@pytest.mark.parametrize(
    "degree, precision",
    [
        (2, 1e-5),
        (3, 1e-5),
        (4, 1e-5),
        (5, 1e-5),
        (10, 1e-5),
        pytest.param(20, 1e-2, marks=pytest.mark.slow),
    ],
)
def test_real_polynomial_has_real_complementary_polynomial(degree: int, precision: float):
    random_state = np.random.RandomState(62)

    for _ in range(10):
        P = random_qsp_polynomial(degree, random_state=random_state, only_real_coeffs=True)
        Q = fast_complementary_polynomial(P, random_state=random_state, only_reals=True)

        np.testing.assert_allclose(np.imag(Q), 0, atol=precision)
        check_gqsp_polynomial_pair_on_random_points_on_unit_circle(
            P, Q, random_state=random_state, rtol=precision
        )


@pytest.mark.parametrize("bitsize", [1, 2])
@pytest.mark.parametrize("negative_power", [0, 1, 2])
@pytest.mark.parametrize("degree, tolerance", [(2, 1e-5), (5, 1e-4)])
def test_generalized_qsp_with_complex_poly_on_random_unitaries(
    bitsize: int, negative_power: int, degree: int, tolerance: float
):
    # TODO Fix high error on degree 20 polynomial
    random_state = np.random.RandomState(42)

    for _ in range(10):
        U = MatrixGate.random(bitsize, random_state=random_state)
        P = random_qsp_polynomial(degree, random_state=random_state)
        Q = fast_complementary_polynomial(P, random_state=random_state)
        verify_generalized_qsp(U, P, negative_power=negative_power, Q=Q, tolerance=tolerance)
