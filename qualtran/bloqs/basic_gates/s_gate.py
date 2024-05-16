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
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import attrs
import numpy as np
from attrs import frozen

from qualtran import Bloq, bloq_example, BloqDocSpec, Register, Signature, SoquetT
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

    The unitary matrix of `cirq.S` is
    $$
    \begin{bmatrix}
        1 & 0 \\
        0 & i 
    \end{bmatrix}
    $$

    Registers:
        q: The qubit
    """
    is_adjoint: bool = False

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(q=1)

    def _t_complexity_(self) -> 'TComplexity':
        return TComplexity(clifford=1)

    def add_my_tensors(
        self,
        tn: 'qtn.TensorNetwork',
        tag: Any,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        import quimb.tensor as qtn

        data = _SMATRIX.conj().T if self.is_adjoint else _SMATRIX
        tn.add(
            qtn.Tensor(
                data=data, inds=(outgoing['q'], incoming['q']), tags=[self.pretty_name(), tag]
            )
        )

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', q: 'CirqQuregT'  # type:ignore[type-var]
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:  # type:ignore[type-var]
        import cirq

        (q,) = q
        return cirq.S(q), {'q': np.array([q])}

    def pretty_name(self) -> str:
        maybe_dag = '†' if self.is_adjoint else ''
        return f'S{maybe_dag}'

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        return TextBox(self.pretty_name())

    def adjoint(self) -> 'Bloq':
        return attrs.evolve(self, is_adjoint=not self.is_adjoint)


@bloq_example
def _s_gate() -> SGate:
    s_gate = SGate()
    return s_gate


_S_GATE_DOC = BloqDocSpec(
    bloq_cls=SGate, import_line='from qualtran.bloqs.basic_gates import SGate', examples=[_s_gate]
)
