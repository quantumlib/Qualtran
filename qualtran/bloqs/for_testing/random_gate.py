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
    bitsize: int
    matrix: Tuple[Tuple[complex, ...], ...] = field(
        converter=lambda mat: tuple(tuple(row) for row in mat)
    )

    @staticmethod
    def create(bitsize: int, *, random_state=None) -> 'RandomGate':
        matrix = random_unitary(2**bitsize, random_state=random_state)
        return RandomGate(bitsize, matrix)

    @property
    def signature(self) -> Signature:
        return Signature.build(q=self.bitsize)

    def _unitary_(self):
        return np.array(self.matrix)

    def adjoint(self) -> 'RandomGate':
        return RandomGate(self.bitsize, np.conj(self.matrix).T)

    def __pow__(self, power):
        if power == -1:
            return self.adjoint()
        return NotImplemented
