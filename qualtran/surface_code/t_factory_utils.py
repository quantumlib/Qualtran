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

import cirq
import numpy as np


def _rotation_unitary(pauli_string: str, theta: float):
    r"""Returns the unitary matrix of e^{j P \theta} where P is the pauli string."""
    num_qubits = len(pauli_string)
    u = cirq.unitary(cirq.DensePauliString(pauli_string))
    return np.cos(theta) * np.eye(1 << num_qubits) + 1j * np.sin(theta) * u


class NoisyPauliRotation(cirq.Gate):
    r"""A channel that applies a pi/8 pauli rotation with possible overshooting to 5pi/8, -pi/8, and 3pi/8.

    The channel is defined as

    $$
        \sum_{k \in \{1, 5, -1, 3\}} p_k e^{-i \frac{\pi}{8} k P} \rho e^{i \frac{\pi}{8} k P}
    $$
    """

    def __init__(self, pauli_string: str, p1: float, p2: float, p3: float):
        """Initializes NoisyPauliRotation.

        Args:
            pauli_string: The pauli string to apply the rotation to.
            p1: The probability of applying the rotation by 5pi/8.
            p2: The probability of applying the rotation by -pi/8.
            p3: The probability of applying the rotation by 3pi/8.
        """
        self._probabilities = [1 - p1 - p2 - p3, p1, p2, p3]
        self._unitaries = [_rotation_unitary(pauli_string, t * np.pi / 8) for t in (1, 5, -1, 3)]
        self.pauli_string = pauli_string

    def _num_qubits_(self):
        return len(self.pauli_string)

    def _mixture_(self):
        return tuple(zip(self._probabilities, self._unitaries))

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=tuple(self.pauli_string))


def storage_error(
    kind: str, probabilities: Sequence[float], qubits: Sequence[cirq.Qid]
) -> Sequence[cirq.Operation]:
    r"""Creates several channels each applying the requested pauli error to a single qubit.

    Each returned operation is a channel that applies the requested error with
    probability `probabilities[i]` to the ith qubit.

    The ith qubit gets transformed as

    $$
        \rho_i \xrightarrow (1 - p_i) \rho_i + p_i E \rho_i E^\dagger
    $$

    where $E$ is the requested error (one of X or Z).

    Args:
        kind: The pauli error to apply, one of 'Z' or 'X'.
        probabilities: The list probabilities of the channels.
        qubits: The qubits to apply the error to.

    Returns:
        A list of operations.

    Raises:
        ValueError: if kind is not 'Z' or 'X'.
    """
    if kind not in ('Z', 'X'):
        raise ValueError(f'kind must be Z or X not {kind}')
    if kind == 'Z':
        return [
            cirq.AsymmetricDepolarizingChannel(p_z=p)(q)
            for p, q in zip(probabilities, qubits, strict=True)
        ]
    return [
        cirq.AsymmetricDepolarizingChannel(p_x=p)(q)
        for p, q in zip(probabilities, qubits, strict=True)
    ]
