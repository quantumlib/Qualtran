#  Copyright 2025 Google LLC
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

from typing import Optional, Sequence

import cirq
import numpy as np


def diamond_norm(choi: np.ndarray) -> float:
    r"""Returns the diamond norm of the map with the given Choi matrix.

    The norm is computed by solving the simplified semidefinite program of Watrous, whose primal form is

    $$
    \max \mathrm{Tr}(Y J) \text{ subject to }
    -I \otimes \sigma \preceq Y \preceq I \otimes \sigma,\ \mathrm{Tr}(\sigma) = 1
    $$

    where $J$ is the Choi matrix and $\sigma$ is a density matrix on the input space. The
    program is solved numerically in double precision, so the result is accurate to the
    tolerance of the solver rather than to the precision of the Choi matrix.

    Args:
        choi: The Choi matrix of the map, using the convention of `cirq.kraus_to_choi` where
            the output space is the first tensor factor.

    Returns:
        The diamond norm of the map.

    Raises:
        ImportError: If cvxpy is not installed.
        ValueError: If the Choi matrix is not square with a square dimension.
        RuntimeError: If the semidefinite program does not solve to optimality.

    References:
        [Simpler semidefinite programs for completely bounded norms](https://arxiv.org/abs/1207.5726)
    """
    if choi.ndim != 2 or choi.shape[0] != choi.shape[1]:
        raise ValueError(f"expected a square Choi matrix, got shape {choi.shape}")
    dim = choi.shape[0]
    d = round(np.sqrt(dim))
    if d * d != dim:
        raise ValueError(f"expected a Choi matrix of square dimension, got dimension {dim}")

    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError(
            "computing the diamond norm requires cvxpy, install it with `pip install cvxpy`"
        ) from exc

    # Symmetrize away numerical imprecision errors, makes cvxpy calculation more consistent
    choi = (choi + choi.conj().T) / 2

    Y = cp.Variable((dim, dim), hermitian=True)
    sigma = cp.Variable((d, d), PSD=True, complex=True)
    bound = cp.kron(np.eye(d), sigma)

    constraints = [-bound << Y, Y << bound, cp.trace(sigma) == 1]

    prob = cp.Problem(cp.Maximize(cp.real(cp.trace(Y @ choi))), constraints)
    prob.solve()

    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"the diamond norm SDP did not solve to optimality: {prob.status}")

    return float(prob.value)


def _qubit_unitary(kraus_list: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    """Returns the single qubit unitary of the channel, or None if it doesn't have one."""
    if len(kraus_list) != 1:
        return None
    matrix = np.asarray(kraus_list[0], dtype=np.complex128)
    if matrix.shape != (2, 2):
        return None
    return matrix


def _unitary_diamond_norm_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Returns the diamond norm distance between two single qubit unitary channels."""
    eigenvalues = np.linalg.eigvals(v.conj().T @ u)
    return float(abs(eigenvalues[1] - eigenvalues[0]))


def diamond_norm_distance(
    kraus_list_a: Sequence[np.ndarray], kraus_list_b: Sequence[np.ndarray]
) -> float:
    """Returns the diamond norm distance between the two channels.

    When both channels are single qubit unitaries the distance is computed analytically,
    otherwise it is computed by solving a semidefinite program, which requires cvxpy.

    Args:
        kraus_list_a: The Kraus operators of the first channel.
        kraus_list_b: The Kraus operators of the second channel.

    Returns:
        The diamond norm distance between the two channels.
    """
    kraus_list_a = [np.asarray(k, dtype=np.complex128) for k in kraus_list_a]
    kraus_list_b = [np.asarray(k, dtype=np.complex128) for k in kraus_list_b]
    u, v = _qubit_unitary(kraus_list_a), _qubit_unitary(kraus_list_b)
    if u is not None and v is not None:
        return _unitary_diamond_norm_distance(u, v)

    choi_difference_matrix = cirq.kraus_to_choi(kraus_list_a) - cirq.kraus_to_choi(kraus_list_b)

    return diamond_norm(choi_difference_matrix)
