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
import scipy.optimize

_PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

# The maximally mixed state and the six axes of the bloch sphere. The maximization is concave so
# a single starting point suffices in theory, the extra ones guard against the objective not
# being differentiable where the eigenvalues of the scaled Choi matrix cross zero.
_STARTING_POINTS = [np.zeros(3)] + [sign * axis for axis in np.eye(3) for sign in (1, -1)]


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


def diamond_norm(choi: np.ndarray) -> float:
    r"""Returns the diamond norm of the single qubit map with the given Choi matrix.

    The map is not required to be a channel, only to be hermiticity preserving, which is the
    case for the difference of two channels. The norm is

    $$
    \|\Delta\|_\diamond = \max_\sigma
    \left\| (I \otimes \sqrt{\sigma}) J (I \otimes \sqrt{\sigma}) \right\|_1
    $$

    where $J$ is the Choi matrix and $\sigma$ is a density matrix on the input space, which for
    a single qubit is a maximization over the three coordinates of the bloch vector of $\sigma$.
    The maximization is concave, being the partial maximization of a linear objective over a
    jointly convex set, so a local maximum is a global one. The maximizing $\sigma$ is not
    always a pure state, a depolarizing channel is for example maximized by the maximally mixed
    state, so the whole bloch ball is searched and not only its surface.

    The maximization is carried out numerically in double precision, so the result is accurate
    to the tolerance of the optimizer rather than to the precision of the Choi matrix.

    Args:
        choi: The Choi matrix of the map, using the convention of `cirq.kraus_to_choi` where
            the output space is the first tensor factor.

    Returns:
        The diamond norm of the map.

    Raises:
        ValueError: If the Choi matrix is not the 4x4 Choi matrix of a single qubit map.
    """
    choi = np.asarray(choi, dtype=np.complex128)
    if choi.shape != (4, 4):
        raise ValueError(
            f"expected the 4x4 Choi matrix of a single qubit map, got shape {choi.shape}"
        )
    # The map is hermiticity preserving, restore the symmetry lost to rounding.
    choi = (choi + choi.conj().T) / 2

    best = 0.0
    for start in _STARTING_POINTS:
        result = scipy.optimize.minimize(
            lambda vector: -_scaled_trace_norm(vector, choi),
            start,
            method="Nelder-Mead",
            options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 5000},
        )
        best = max(best, -float(result.fun))
    return best


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
    """Returns the diamond norm distance between the two single qubit channels.

    When both channels are unitaries the distance is computed analytically, otherwise it is
    computed by `diamond_norm` of the difference of the Choi matrices of the channels.

    Args:
        kraus_list_a: The Kraus operators of the first channel.
        kraus_list_b: The Kraus operators of the second channel.

    Returns:
        The diamond norm distance between the two channels.

    Raises:
        ValueError: If the channels are not single qubit channels.
    """
    kraus_list_a = [np.asarray(k, dtype=np.complex128) for k in kraus_list_a]
    kraus_list_b = [np.asarray(k, dtype=np.complex128) for k in kraus_list_b]
    u, v = _qubit_unitary(kraus_list_a), _qubit_unitary(kraus_list_b)
    if u is not None and v is not None:
        return _unitary_diamond_norm_distance(u, v)

    choi_difference_matrix = cirq.kraus_to_choi(kraus_list_a) - cirq.kraus_to_choi(kraus_list_b)

    return diamond_norm(choi_difference_matrix)
