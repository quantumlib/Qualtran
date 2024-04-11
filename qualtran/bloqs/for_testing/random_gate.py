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
from typing import Tuple

import numpy as np
from attrs import field, frozen
from cirq.testing import random_unitary

from qualtran import GateWithRegisters, Signature


@frozen
class RandomGate(GateWithRegisters):
    """Gate with a uniformly random unitary action

    Args:
        bitsize: number of qubits $n$
        matrix: a $2^n \times 2^n$ complex unitary matrix

    Registers:
        q: $n$-qubit register
    """

    bitsize: int
    matrix: Tuple[Tuple[complex, ...], ...] = field(
        converter=lambda mat: tuple(tuple(row) for row in mat)
    )

    def __attrs_post_init__(self):
        # verify that matrix is unitary
        matrix = np.array(self.matrix)
        np.testing.assert_allclose(matrix @ matrix.conj().T, np.eye(2**self.bitsize), atol=1e-10)
        np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(2**self.bitsize), atol=1e-10)

    @property
    def signature(self) -> Signature:
        return Signature.build(q=self.bitsize)

    @classmethod
    def create(cls, bitsize: int, *, random_state=None) -> 'RandomGate':
        """generate a uniformly random unitary on `bitsize` qubits"""
        matrix = random_unitary(2**bitsize, random_state=random_state)
        return cls(bitsize, matrix)

    def _unitary_(self):
        return np.array(self.matrix)

    def adjoint(self) -> 'RandomGate':
        return RandomGate(self.bitsize, np.conj(self.matrix).T)

    def __pow__(self, power):
        if power == -1:
            return self.adjoint()
        return NotImplemented
