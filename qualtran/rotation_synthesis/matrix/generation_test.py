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

import cirq
import numpy as np

from qualtran.rotation_synthesis.matrix import generate_cliffords, generate_rotations


def _are_close_up_to_global_phase(u, v):
    i, j = np.unravel_index(  # pylint: disable=unbalanced-tuple-unpacking
        np.abs(u).argmax(), u.shape
    )
    return np.allclose(u * v[i, j] / u[i, j], v)


def test_generated_rotations_determinant():
    all_rotations = generate_rotations(5)
    for n in range(len(all_rotations)):
        assert np.allclose(
            [abs(np.linalg.det(r.matrix.astype(complex))) for r in all_rotations[n]],
            2 * (2 + np.sqrt(2)) ** n,
        )


_SQRT2 = np.sqrt(2)
_X = np.array([[0, 1], [1, 0]])
_Y = np.array([[0, -1j], [1j, 0]])
_Z = np.array([[1, 0], [0, -1]])
_TX_numpy = (np.eye(2) + (np.eye(2) - 1j * _X) / _SQRT2) / np.sqrt(2 + _SQRT2)
_TY_numpy = (np.eye(2) + (np.eye(2) - 1j * _Y) / _SQRT2) / np.sqrt(2 + _SQRT2)
_TZ_numpy = (np.eye(2) + (np.eye(2) - 1j * _Z) / _SQRT2) / np.sqrt(2 + _SQRT2)

_numpy_matrix_for_symbol = {
    "H": np.array([[1, 1], [1, -1]]) / np.sqrt(2),
    "S": np.array([[1, 0], [0, 1j]]),
    "Tx": _TX_numpy,
    "Ty": _TY_numpy,
    "Tz": _TZ_numpy,
}


def test_generated_rotations_unitary():
    all_rotations = generate_rotations(5)
    for row in all_rotations:
        for r in row:
            u = np.sqrt(2) * np.eye(2, dtype=np.complex128)
            gates = r.gates
            assert gates is not None
            for p in gates:
                u = _numpy_matrix_for_symbol[p] @ u
            assert _are_close_up_to_global_phase(u, r.numpy())


def test_generate_cliffords():
    cliffords = generate_cliffords()
    cirq_cliffords = [
        cirq.unitary(c) for c in cirq.SingleQubitCliffordGate.all_single_qubit_cliffords
    ]
    assert np.allclose(np.abs([np.linalg.det(c.numpy()) for c in cliffords]), 2)
    sqrt2 = np.sqrt(2)
    for c in cliffords:
        u = c.numpy() / sqrt2
        assert np.any([_are_close_up_to_global_phase(u, c) for c in cirq_cliffords])
