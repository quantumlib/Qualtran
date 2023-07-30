#  Copyright 2023 Google LLC
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

from functools import cached_property
from typing import Dict, Tuple

import cirq
from attrs import frozen

from qualtran import Bloq, Signature
from qualtran.cirq_interop import CirqQuregT


@frozen
class Hadamard(Bloq):
    """The Hadamard gate."""

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(q=1)

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', q: 'CirqQuregT'
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        import cirq

        (q,) = q
        return cirq.H(q), {'q': [q]}


@frozen
class MultiHadamard(Bloq):
    """The Hadamard gate on bitsize qubits"""

    bitsize: int

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(q=self.bitsize)

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', q: 'CirqQuregT'
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        import cirq

        (q,) = q
        return cirq.H.on_each(q), {'q': [q]}
