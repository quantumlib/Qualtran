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

import cirq
import numpy as np
import pytest

import qualtran.rotation_synthesis as rs
import qualtran.rotation_synthesis.channels as ch

# diamond_norm_distance falls back to qubit_diamond_norm_lower_bound when cvxpy is missing,
# only a test that calls diamond_norm itself needs the solver.
requires_cvxpy = pytest.mark.skipif(
    importlib.util.find_spec("cvxpy") is None, reason="requires cvxpy"
)

# The solver works in double precision, its answers are accurate to a few significant digits.
SOLVER_ATOL = 1e-5

X = cirq.unitary(cirq.X)
Y = cirq.unitary(cirq.Y)
Z = cirq.unitary(cirq.Z)
I = np.eye(2)


def _rz(theta: float) -> np.ndarray:
    r"""Returns the matrix of $e^{i \theta Z}$."""
    return np.diag([np.exp(1j * theta), np.exp(-1j * theta)])


def _dephasing(p: float) -> list[np.ndarray]:
    return [np.sqrt(1 - p) * I, np.sqrt(p) * Z]


def _depolarizing(p: float) -> list[np.ndarray]:
    return [np.sqrt(1 - 3 * p / 4) * I] + [np.sqrt(p / 4) * pauli for pauli in (X, Y, Z)]


def _amplitude_damping(gamma: float) -> list[np.ndarray]:
    return [np.array([[1, 0], [0, np.sqrt(1 - gamma)]]), np.array([[0, np.sqrt(gamma)], [0, 0]])]


@pytest.mark.parametrize("delta", [1e-1, 1e-3, 1e-6, 1e-9])
def test_unitary_distance_matches_analytical_formula(delta):
    # The distance between two Z rotations differing by delta is 2|sin(delta)|. The eigenvalues
    # are computed in double precision, which limits the accuracy for a tiny delta.
    distance = ch.diamond_norm_distance([_rz(0.3 + delta)], [_rz(0.3)])
    np.testing.assert_allclose(distance, 2 * abs(np.sin(delta)), rtol=1e-6)


def test_unitary_distance_to_itself_is_zero():
    u = cirq.unitary(cirq.T)
    assert ch.diamond_norm_distance([u], [u]) == pytest.approx(0, abs=1e-12)


def test_unitary_distance_ignores_global_phase():
    u = cirq.unitary(cirq.H)
    assert ch.diamond_norm_distance([u], [np.exp(1.3j) * u]) == pytest.approx(0, abs=1e-12)


def test_unitary_distance_is_symmetric():
    u, v = cirq.unitary(cirq.H), _rz(0.4)
    assert ch.diamond_norm_distance([u], [v]) == pytest.approx(ch.diamond_norm_distance([v], [u]))


def test_orthogonal_unitaries_are_maximally_distant():
    # X and Z map |0> to orthogonal states, so the channels are perfectly distinguishable.
    assert ch.diamond_norm_distance([X], [Z]) == pytest.approx(2, abs=1e-12)


@pytest.mark.parametrize("choi", [np.zeros((3, 3)), np.zeros((4, 5)), np.zeros(4)])
def test_diamond_norm_rejects_invalid_choi_matrix(choi):
    with pytest.raises(ValueError):
        ch.diamond_norm(choi)


@pytest.mark.parametrize(
    ["kraus", "expected"],
    [
        (_dephasing(0.05), 2 * 0.05),
        (_dephasing(0.25), 2 * 0.25),
        (_depolarizing(0.1), 3 * 0.1 / 2),
        (_depolarizing(0.4), 3 * 0.4 / 2),
        # Amplitude damping is not symmetric between the input and the output space, so it also
        # pins down the tensor factor ordering of the Choi matrix.
        (_amplitude_damping(0.05), 2 * 0.05),
        (_amplitude_damping(0.3), 2 * 0.3),
    ],
)
def test_distance_to_identity_of_known_channels(kraus, expected):
    distance = ch.diamond_norm_distance(kraus, [I])
    np.testing.assert_allclose(distance, expected, atol=SOLVER_ATOL)


@requires_cvxpy
@pytest.mark.parametrize("seed", range(3))
def test_semidefinite_program_agrees_with_unitary_formula(seed):
    rng = np.random.default_rng(seed)
    unitaries = []
    for _ in range(2):
        matrix = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        q, r = np.linalg.qr(matrix)
        unitaries.append(q @ np.diag(np.diag(r) / abs(np.diag(r))))
    u, v = unitaries

    # diamond_norm_distance takes the analytical shortcut for unitaries, call the solver directly.
    choi = cirq.kraus_to_choi([u]) - cirq.kraus_to_choi([v])
    np.testing.assert_allclose(
        ch.diamond_norm(choi), ch.diamond_norm_distance([u], [v]), atol=SOLVER_ATOL
    )


def test_distance_between_mixtures_of_unitaries():
    # A mixture of U and V is at most as far from U as V is.
    u, v = _rz(0.3), _rz(0.5)
    mixture = [np.sqrt(0.25) * u, np.sqrt(0.75) * v]
    distance = ch.diamond_norm_distance(mixture, [u])
    assert 0 < distance < ch.diamond_norm_distance([v], [u])


def test_channel_distance_matches_unitary_method():
    config = rs.NumpyConfig
    a = ch.UnitaryChannel.from_sequence(["H", "Tz", "S"])
    b = ch.UnitaryChannel.from_sequence(["H", "Tz", "Tx"])
    expected = a.diamond_norm_distance_to_unitary(b.to_matrix().numpy(config), config)
    np.testing.assert_allclose(a.diamond_norm_distance_to_channel(b, config), expected, rtol=1e-9)


def test_mixed_diagonal_protocol_matches_analytical_distance():
    config = rs.with_dps(100)
    theta = 0.1
    channel = rs.mixed_diagonal_protocol(theta, 1e-6, max_n=200, config=config)
    assert channel is not None
    expected = float(channel.diamond_norm_distance_to_rz(theta, config))
    distance = ch.diamond_norm_distance(channel.kraus(config), [_rz(theta)])
    np.testing.assert_allclose(distance, expected, rtol=1e-3)
