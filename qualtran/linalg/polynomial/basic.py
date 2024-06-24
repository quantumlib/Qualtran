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
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


def evaluate_polynomial_of_unitary_matrix(
    P: Sequence[complex], U: NDArray, *, offset: int = 0
) -> NDArray:
    r"""Computes $U^{\mathsf{offset}} P(U)$ for a polynomial P and unitary U.

    Args:
        P: Coefficients of a complex polynomial (P[i] is the coefficient of $x^i$)
        U: Unitary matrix
        offset: the power offset $U^\mathsf{offset}$ to multiply the result by.
    """
    assert U.ndim == 2 and U.shape[0] == U.shape[1]

    if offset < 0:
        pow_U = np.linalg.matrix_power(U.conj().T, -offset)
    else:
        pow_U = np.linalg.matrix_power(U, offset)

    result = np.zeros(U.shape, dtype=U.dtype)

    for c in P:
        result += pow_U * c
        pow_U = pow_U @ U

    return result
