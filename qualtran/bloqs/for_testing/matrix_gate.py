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
from typing import Dict, List, Tuple, TYPE_CHECKING

import numpy as np
from attrs import field, frozen

from qualtran import ConnectionT, GateWithRegisters, Signature

if TYPE_CHECKING:
    import quimb.tensor as qtn


@frozen
class MatrixGate(GateWithRegisters):
    """A unitary n-qubit defined entirely by its numpy matrix.

    Args:
        bitsize: number of qubits $n$
        matrix: a $2^n \times 2^n$ complex unitary matrix
        atol: tolerance to use for unitarity checking

    Registers:
        q: $n$-qubit register
    """

    bitsize: int
    matrix: Tuple[Tuple[complex, ...], ...] = field(
        converter=lambda mat: tuple(tuple(row) for row in mat)
    )
    atol: float = 1e-10

    def __attrs_post_init__(self):
        # verify that matrix is unitary
        matrix = np.array(self.matrix)
        np.testing.assert_allclose(
            matrix @ matrix.conj().T, np.eye(2**self.bitsize), atol=self.atol
        )
        np.testing.assert_allclose(
            matrix.conj().T @ matrix, np.eye(2**self.bitsize), atol=self.atol
        )

    @property
    def signature(self) -> Signature:
        return Signature.build(q=self.bitsize)

    @classmethod
    def random(cls, bitsize: int, *, random_state=None) -> 'MatrixGate':
        """generate a uniformly random unitary on `bitsize` qubits"""
        from cirq.testing import random_unitary

        matrix = random_unitary(2**bitsize, random_state=random_state)
        return cls(bitsize, matrix)

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        import quimb.tensor as qtn

        data = np.array(self.matrix).reshape((2,) * (self.bitsize * 2))
        return [
            qtn.Tensor(
                data=data,
                inds=[(outgoing['q'], j) for j in range(self.bitsize)]
                + [(incoming['q'], j) for j in range(self.bitsize)],
                tags=[str(self)],
            )
        ]

    def _unitary_(self):
        return np.array(self.matrix)

    def adjoint(self) -> 'MatrixGate':
        return MatrixGate(self.bitsize, np.conj(self.matrix).T, self.atol)
