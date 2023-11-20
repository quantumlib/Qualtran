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
from typing import Dict, Tuple, TYPE_CHECKING

import numpy as np
from attrs import frozen

from qualtran import Bloq, bloq_example, CompositeBloq, DecomposeTypeError, Signature, SoquetT
from qualtran.cirq_interop.t_complexity_protocol import TComplexity

if TYPE_CHECKING:
    import cirq
    import quimb.tensor as qtn

    from qualtran.cirq_interop import CirqQuregT

_HADAMARD = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)


@frozen
class Hadamard(Bloq):
    r"""The Hadamard gate

    This converts between the X and Z basis.

    $$
    H |0\rangle = |+\rangle \\
    H |-\rangle = |1\rangle
    $$

    Registers:
        q: The qubit
    """

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(q=1)

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic")

    def add_my_tensors(
        self,
        tn: 'qtn.TensorNetwork',
        binst,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        import quimb.tensor as qtn

        tn.add(
            qtn.Tensor(
                data=_HADAMARD, inds=(outgoing['q'], incoming['q']), tags=[self.short_name(), binst]
            )
        )

    def short_name(self) -> 'str':
        return 'H'

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', q: 'CirqQuregT'
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        import cirq

        (q,) = q
        return cirq.H(q), {'q': np.array([q])}

    def t_complexity(self):
        return TComplexity(clifford=1)


@bloq_example
def _hadamard() -> Hadamard:
    hadamard = Hadamard()
    return hadamard
