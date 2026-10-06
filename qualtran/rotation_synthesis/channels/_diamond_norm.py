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

import importlib.util
from typing import Optional, Sequence

import cirq
import numpy as np
import scipy.optimize

_PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


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

    The distance is computed with the cheapest method that applies:
    - If both channels are single qubit unitaries it is computed analytically.
    - Otherwise, if cvxpy is installed, `diamond_norm` solves a semidefinite program.
    - Otherwise, for a single qubit map, `qubit_diamond_norm_lower_bound` maximizes over the
      input density matrix, which is slower and only gives a lower bound on the distance.

    Args:
        kraus_list_a: The Kraus operators of the first channel.
        kraus_list_b: The Kraus operators of the second channel.

    Returns:
        The diamond norm distance between the two channels.

    Raises:
        ImportError: If cvxpy is not installed and the channels are not single qubit channels.
    """
    kraus_list_a = [np.asarray(k, dtype=np.complex128) for k in kraus_list_a]
    kraus_list_b = [np.asarray(k, dtype=np.complex128) for k in kraus_list_b]
    u, v = _qubit_unitary(kraus_list_a), _qubit_unitary(kraus_list_b)
    if u is not None and v is not None:
        return _unitary_diamond_norm_distance(u, v)

    choi_difference_matrix = cirq.kraus_to_choi(kraus_list_a) - cirq.kraus_to_choi(kraus_list_b)

    if importlib.util.find_spec("cvxpy") is None and choi_difference_matrix.shape == (4, 4):
        return qubit_diamond_norm_lower_bound(choi_difference_matrix)

    return diamond_norm(choi_difference_matrix)


def _sqrt_density_matrix(bloch_vector: np.ndarray) -> np.ndarray:
    r"""Returns $\sqrt{\sigma}$ of the qubit density matrix with the given bloch vector.

    The density matrix $\sigma = (I + \vec{r} \cdot \vec{\sigma})/2$ has eigenvalues
    $(1 \pm |\vec{r}|)/2$, so its square root is $a I + b\, \hat{r} \cdot \vec{\sigma}$ where
    $a$ and $b$ are the half sum and the half difference of the square roots of the
    eigenvalues. A vector outside the bloch ball is projected onto its surface.
    """
    radius = float(np.linalg.norm(bloch_vector))
    if radius > 1:
        bloch_vector = bloch_vector / radius
        radius = 1.0
    sqrt_plus, sqrt_minus = np.sqrt((1 + radius) / 2), np.sqrt((1 - radius) / 2)
    # (sqrt_plus - sqrt_minus) / 2 times the unit vector, written to avoid dividing by zero.
    scale = 0.0 if radius < 1e-15 else (sqrt_plus - sqrt_minus) / (2 * radius)
    return (sqrt_plus + sqrt_minus) / 2 * np.eye(2) + scale * (
        bloch_vector[0] * _PAULI_X + bloch_vector[1] * _PAULI_Y + bloch_vector[2] * _PAULI_Z
    )


def _scaled_trace_norm(bloch_vector: np.ndarray, choi: np.ndarray) -> float:
    r"""Returns $\|(I \otimes \sqrt{\sigma}) J (I \otimes \sqrt{\sigma})\|_1$."""
    root = np.kron(np.eye(2), _sqrt_density_matrix(bloch_vector))
    scaled = root @ choi @ root
    return float(np.abs(np.linalg.eigvalsh((scaled + scaled.conj().T) / 2)).sum())


def qubit_diamond_norm_lower_bound(
    choi: np.ndarray, num_random_starts: int = 16, seed: int = 0
) -> float:
    r"""Returns a lower bound on the diamond norm of a single qubit map.

    Eliminating the matrix variable of the semidefinite program solved by `diamond_norm`
    analytically leaves a maximization over the input density matrix alone

    $$
    \|\Delta\|_\diamond = \max_\sigma
    \left\| (I \otimes \sqrt{\sigma}) J (I \otimes \sqrt{\sigma}) \right\|_1
    $$

    which for a single qubit is a maximization over the three coordinates of the bloch vector
    of $\sigma$. Unlike `diamond_norm` this needs no semidefinite program solver, but the
    maximization is not known to be concave, so a local optimizer only gives a lower bound on
    the norm. The maximizing $\sigma$ is not always a pure state, for example a depolarizing
    channel is maximized by the maximally mixed state, so the whole bloch ball is searched and
    not only its surface.

    Args:
        choi: The Choi matrix of the map, using the convention of `cirq.kraus_to_choi` where
            the output space is the first tensor factor.
        num_random_starts: The number of random starting points used in addition to the
            deterministic ones.
        seed: The seed of the random starting points.

    Returns:
        A lower bound on the diamond norm of the map, which is the norm itself whenever the
        maximization finds a global maximum.

    Raises:
        ValueError: If the Choi matrix is not the 4x4 Choi matrix of a single qubit map.
    """
    if choi.shape != (4, 4):
        raise ValueError(f"expected the 4x4 choi matrix of a qubit map, got shape {choi.shape}")
    choi = np.asarray(choi, dtype=np.complex128)
    choi = (choi + choi.conj().T) / 2

    # The maximally mixed state and the six axes of the bloch sphere, plus random points.
    starts = [np.zeros(3)]
    starts.extend(sign * axis for axis in np.eye(3) for sign in (1, -1))
    rng = np.random.default_rng(seed)
    for _ in range(num_random_starts):
        direction = rng.normal(size=3)
        starts.append(direction / np.linalg.norm(direction) * rng.uniform(0, 1))

    best = 0.0
    for start in starts:
        result = scipy.optimize.minimize(
            lambda vector: -_scaled_trace_norm(vector, choi),
            start,
            method="Nelder-Mead",
            options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 5000},
        )
        best = max(best, -float(result.fun))
    return best
