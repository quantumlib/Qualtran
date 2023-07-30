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

from functools import cached_property, reduce
from typing import Dict, Tuple

import cirq
import numpy as np
import quimb.tensor as qtn
from attrs import frozen

from qualtran import Bloq, Signature, SoquetT
from qualtran.cirq_interop import CirqQuregT

_HADAMARD = np.array([[1.0, 1.0], [1.0, -1.0]]) / (2**0.5)


@frozen
class Hadamard(Bloq):
    """The Hadamard gate.

    Args:
        bitsize: The number qubits to add a Hadamard gate to.

    Registers:
     - q: bitsize-bit register.
    """

    bitsize: int = 1

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(q=self.bitsize)

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', q: 'CirqQuregT'
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        import cirq

        if len(q) == 1:
            (q,) = q
            return cirq.H(q), {'q': [q]}
        else:
            return cirq.H.on_each(q), {'q': q}

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        binst,
        *,
        incoming: Dict[str, SoquetT],
        outgoing: Dict[str, SoquetT],
    ):
        tn.add(
            qtn.Tensor(
                data=reduce(np.kron, (_HADAMARD,) * self.bitsize),
                inds=(outgoing['q'], incoming['q']),
                tags=[self.short_name(), binst],
            )
        )
