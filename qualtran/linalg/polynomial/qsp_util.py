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
from typing import Sequence, Union

import numpy as np
from numpy.polynomial import Polynomial
from numpy.typing import NDArray

from qualtran.symbolics import Shaped


def check_polynomial_pair_on_random_points_on_unit_circle(
    P: Union[Sequence[complex], Polynomial, Shaped],
    Q: Union[Sequence[complex], Polynomial, Shaped],
    *,
    random_state: np.random.RandomState,
    rtol: float = 1e-7,
    n_points: int = 1000,
):
    P = Polynomial(P)
    Q = Polynomial(Q)

    for _ in range(n_points):
        z = np.exp(random_state.random() * np.pi * 2j)
        np.testing.assert_allclose(np.abs(P(z)) ** 2 + np.abs(Q(z)) ** 2, 1, rtol=rtol)


def random_qsp_polynomial(
    degree: int, *, random_state: np.random.RandomState, only_real_coeffs=False
) -> Sequence[complex]:
    poly = random_state.random(size=degree) / degree
    if not only_real_coeffs:
        poly = poly * np.exp(random_state.random(size=degree) * np.pi * 2j)
    return list(poly)


def evaluate_polynomial_of_matrix(
    P: Sequence[complex], U: NDArray, *, negative_power: int = 0
) -> NDArray:
    assert U.ndim == 2 and U.shape[0] == U.shape[1]

    pow_U = np.linalg.matrix_power(U.conj().T, negative_power)
    result = np.zeros(U.shape, dtype=U.dtype)

    for c in P:
        result += pow_U * c
        pow_U = pow_U @ U

    return result
