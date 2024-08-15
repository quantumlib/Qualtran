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
from typing import TYPE_CHECKING, Union

import numpy as np
from attrs import frozen

from qualtran import Bloq, DecomposeTypeError, QBit, Register, Side, Signature

if TYPE_CHECKING:
    import cirq

    from qualtran.cirq_interop import CirqQuregT


@frozen
class Measure(Bloq):
    """Single qubit measurement gate."""

    @property
    def signature(self) -> 'Signature':
        return Signature([Register("q", QBit(), side=Side.LEFT)])

    def decompose_bloq(self):
        raise DecomposeTypeError(f"{self} is atomic")

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', q: 'CirqQuregT'
    ) -> tuple[Union['cirq.Operation', None], dict[str, 'CirqQuregT']]:
        import cirq

        measure = cirq.MeasurementGate(num_qubits=1)

        (q,) = q
        return measure(q), {'q': np.array([q])}
