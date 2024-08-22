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
from cirq.testing.lin_alg_utils import random_orthogonal
from numpy.typing import NDArray


def random_hermitian_matrix(dim: int, random_state: np.random.RandomState) -> NDArray:
    """Return a random Hermitian matrix of given dimension and norm <= 1."""
    a = random_orthogonal(dim, random_state=random_state)
    return (a + a.conj().T) / 2


def unitary_distance_ignoring_global_phase(U: NDArray, V: NDArray) -> float:
    r"""Distance between two unitaries ignoring global phase.

    Given two $N \times N$ unitaries $U, V$, their distance is given by (Eq. 1 of Ref. [1]):

        $$
            \sqrt{\frac{N - \abs{\tr(U^\dagger V)}}{N}}
        $$

    Args:
        U: N x N unitary matrix
        V: N x N unitary matrix

    References:
        [Constructing arbitrary Steane code single logical qubit fault-tolerant gates](https://arxiv.org/abs/quant-ph/0411206)
        Fowler. 2010. Section 1, Eq 1.
    """
    if U.shape != V.shape:
        raise ValueError(f"Inputs must have same dimension, got {U.shape}, {V.shape}")
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError(f"Inputs must be square matrices, got shape {U.shape}")

    N = U.shape[0]
    trace = np.trace(U.conj().T @ V)
    d_squared = 1 - np.abs(trace) / N
    return np.sqrt(max(0, d_squared))
