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

import cirq
import numpy as np
import pytest

from qualtran.surface_code.t_factory_utils import NoisyPauliRotation, storage_error


def test_storage_error():
    qs = cirq.LineQubit.range(2)
    assert storage_error('Z', [0.1, 0.3], qs) == [
        cirq.AsymmetricDepolarizingChannel(p_z=0.1)(qs[0]),
        cirq.AsymmetricDepolarizingChannel(p_z=0.3)(qs[1]),
    ]

    assert storage_error('X', [0.2, 0.7], qs) == [
        cirq.AsymmetricDepolarizingChannel(p_x=0.2)(qs[0]),
        cirq.AsymmetricDepolarizingChannel(p_x=0.7)(qs[1]),
    ]

    with pytest.raises(ValueError):
        _ = storage_error('Y', [0.1, 0.3], qs)


def test_pauli_channel():
    probabilities = [0.4, 0.1, 0.2, 0.3]
    angles = [t * np.pi / 8 for t in (1, 5, -1, 3)]
    u = cirq.unitary(cirq.DensePauliString('ZIZZ'))
    desired_kraus = [
        np.sqrt(p) * (np.cos(t) * np.eye(16) + 1j * np.sin(t) * u)
        for t, p in zip(angles, probabilities)
    ]
    p = NoisyPauliRotation('ZIZZ', *probabilities[1:])
    np.testing.assert_allclose(cirq.kraus(p), desired_kraus)

    cirq.testing.assert_has_diagram(
        cirq.Circuit(p(*cirq.LineQubit.range(4))),
        """
0: ───Z───
      │
1: ───I───
      │
2: ───Z───
      │
3: ───Z───
""",
    )
