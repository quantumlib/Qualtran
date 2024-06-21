#  Copyright 2024 Google LLC
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

from qualtran.linalg.polynomial.jacobi_anger_approximations import (
    approx_exp_cos_by_jacobi_anger,
    approx_exp_sin_by_jacobi_anger,
    degree_jacobi_anger_approximation,
)


@pytest.mark.parametrize("t", [2, 3, 5, 10])
@pytest.mark.parametrize("precision", [1e-5, 1e-7, 1e-10])
def test_exp_cos_approximation(t: float, precision: float):
    random_state = np.random.RandomState(42 + int(t))

    degree = degree_jacobi_anger_approximation(t, precision=precision)
    assert isinstance(degree, int)
    P = np.polynomial.Polynomial(approx_exp_cos_by_jacobi_anger(t, degree=degree))
    theta = 2 * np.pi * random_state.random(1000)
    e_itheta = np.exp(1j * theta)
    np.testing.assert_allclose(
        P(e_itheta) * e_itheta ** (-degree), np.exp(1j * t * np.cos(theta)), rtol=precision * 2
    )


@pytest.mark.parametrize("t", [2, 3, 5, 10])
@pytest.mark.parametrize("precision", [1e-5, 1e-7, 1e-10])
def test_exp_sin_approximation(t: float, precision: float):
    random_state = np.random.RandomState(42 + int(t))

    degree = degree_jacobi_anger_approximation(t, precision=precision)
    assert isinstance(degree, int)
    P = np.polynomial.Polynomial(approx_exp_sin_by_jacobi_anger(t, degree=degree))
    theta = 2 * np.pi * random_state.random(1000)
    e_itheta = np.exp(1j * theta)
    np.testing.assert_allclose(
        P(e_itheta) * e_itheta ** (-degree), np.exp(1j * t * np.sin(theta)), rtol=precision * 2
    )
