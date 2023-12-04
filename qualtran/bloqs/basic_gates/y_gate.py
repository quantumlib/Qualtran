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
from typing import Any, Dict, Tuple, TYPE_CHECKING

import numpy as np
import quimb.tensor as qtn
from attrs import frozen

from qualtran import Bloq, Signature, SoquetT

if TYPE_CHECKING:
    import cirq

    from qualtran.cirq_interop import CirqQuregT

_PAULIY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)


@frozen
class YGate(Bloq):
    """The Pauli Y gate.

    This causes a bit flip (with a complex phase): Y|0> = 1j|1>, Y|1> = -1j|0>
    """

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(q=1)

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        tag: Any,
        *,
        incoming: Dict[str, SoquetT],
        outgoing: Dict[str, SoquetT],
    ):
        tn.add(
            qtn.Tensor(
                data=_PAULIY, inds=(outgoing['q'], incoming['q']), tags=[self.short_name(), tag]
            )
        )

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', q: 'CirqQuregT'
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        import cirq

        (q,) = q
        return cirq.Y(q), {'q': [q]}
