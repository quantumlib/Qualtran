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
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import attrs
import numpy as np
from attrs import frozen

from qualtran import Bloq, bloq_example, BloqDocSpec, ConnectionT, Register, Signature
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.drawing import Text, TextBox, WireSymbol

if TYPE_CHECKING:
    import cirq
    import quimb.tensor as qtn

    from qualtran.cirq_interop import CirqQuregT


_SMATRIX = np.array([[1, 0], [0, 1j]], dtype=np.complex128)


@frozen
class SGate(Bloq):
    r"""The S gate.

    The unitary matrix of `SGate` is
    $$
    \begin{bmatrix}
        1 & 0 \\
        0 & i 
    \end{bmatrix}
    $$

    It is the 'square root' of the Z gate: $S\cdot S = Z$.

    Registers:
        q: The qubit
    """
    is_adjoint: bool = False

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(q=1)

    def _t_complexity_(self) -> 'TComplexity':
        return TComplexity(clifford=1)

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        import quimb.tensor as qtn

        data = _SMATRIX.conj().T if self.is_adjoint else _SMATRIX
        return [
            qtn.Tensor(data=data, inds=[(outgoing['q'], 0), (incoming['q'], 0)], tags=[str(self)])
        ]

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', q: 'CirqQuregT'  # type:ignore[type-var]
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:  # type:ignore[type-var]
        import cirq

        (q,) = q
        p = -1 if self.is_adjoint else 1
        return cirq.S(q) ** p, {'q': np.array([q])}

    def __str__(self) -> str:
        maybe_dag = '†' if self.is_adjoint else ''
        return f'S{maybe_dag}'

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        return TextBox(str(self))

    def adjoint(self) -> 'Bloq':
        return attrs.evolve(self, is_adjoint=not self.is_adjoint)


@bloq_example
def _s_gate() -> SGate:
    s_gate = SGate()
    return s_gate


_S_GATE_DOC = BloqDocSpec(bloq_cls=SGate, examples=[_s_gate], call_graph_example=None)
