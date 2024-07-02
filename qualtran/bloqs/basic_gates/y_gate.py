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
from typing import Dict, List, Tuple, TYPE_CHECKING

import numpy as np
from attrs import frozen

from qualtran import Bloq, ConnectionT, Signature
from qualtran.cirq_interop.t_complexity_protocol import TComplexity

if TYPE_CHECKING:
    import cirq
    import quimb.tensor as qtn

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

    def adjoint(self) -> 'Bloq':
        return self

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        import quimb.tensor as qtn

        return [
            qtn.Tensor(
                data=_PAULIY, inds=[(outgoing['q'], 0), (incoming['q'], 0)], tags=[str(self)]
            )
        ]

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', q: 'CirqQuregT'
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        import cirq

        (q,) = q
        return cirq.Y(q), {'q': np.asarray([q])}

    def _t_complexity_(self) -> 'TComplexity':
        return TComplexity(clifford=1)
