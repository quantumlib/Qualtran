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
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
from attrs import frozen

from qualtran import (
    AddControlledT,
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    CompositeBloq,
    CtrlSpec,
    DecomposeTypeError,
    Register,
    Signature,
    SoquetT,
)
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.drawing import Circle, Text, TextBox, WireSymbol

if TYPE_CHECKING:
    import cirq
    import quimb.tensor as qtn

    from qualtran.cirq_interop import CirqQuregT

_HADAMARD = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)


@frozen
class Hadamard(Bloq):
    r"""The Hadamard gate

    This converts between the X and Z basis.

    $$\begin{aligned}
    H |0\rangle = |+\rangle \\
    H |-\rangle = |1\rangle
    \end{aligned}$$

    Registers:
        q: The qubit
    """

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(q=1)

    def adjoint(self) -> 'Bloq':
        return self

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic")

    def add_my_tensors(
        self,
        tn: 'qtn.TensorNetwork',
        tag: Any,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        import quimb.tensor as qtn

        tn.add(qtn.Tensor(data=_HADAMARD, inds=(outgoing['q'], incoming['q']), tags=["H", tag]))

    def get_ctrl_system(
        self, ctrl_spec: Optional['CtrlSpec'] = None
    ) -> Tuple['Bloq', 'AddControlledT']:
        if not (ctrl_spec is None or ctrl_spec == CtrlSpec()):
            return super().get_ctrl_system(ctrl_spec=ctrl_spec)

        bloq = CHadamard()

        def _add_ctrled(
            bb: 'BloqBuilder', ctrl_soqs: Sequence['SoquetT'], in_soqs: Dict[str, 'SoquetT']
        ) -> Tuple[Iterable['SoquetT'], Iterable['SoquetT']]:
            (ctrl,) = ctrl_soqs
            ctrl, q = bb.add(bloq, ctrl=ctrl, q=in_soqs['q'])
            return ((ctrl,), (q,))

        return bloq, _add_ctrled

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', q: 'CirqQuregT'  # type: ignore[type-var]
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:  # type: ignore[type-var]
        import cirq

        (q,) = q
        return cirq.H(q), {'q': np.array([q])}

    def _t_complexity_(self):
        return TComplexity(clifford=1)

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        return TextBox('H')

    def __str__(self):
        return 'H'


@bloq_example
def _hadamard() -> Hadamard:
    hadamard = Hadamard()
    return hadamard


_HADAMARD_DOC = BloqDocSpec(bloq_cls=Hadamard, examples=[_hadamard], call_graph_example=None)


@frozen
class CHadamard(Bloq):
    r"""The controlled Hadamard gate

    Registers:
        ctrl: The control qubit.
        q: The target qubit.
    """

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(ctrl=1, q=1)

    def adjoint(self) -> 'Bloq':
        return self

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic")

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        if reg.name == 'ctrl':
            return Circle()
        if reg.name == 'q':
            return TextBox('H')
        raise ValueError(f"Unknown register {reg}")

    def __str__(self):
        return 'CH'


@bloq_example
def _chadamard() -> CHadamard:
    chadamard = Hadamard().controlled()
    return chadamard


_CHADAMARD_DOC = BloqDocSpec(bloq_cls=CHadamard, examples=[_chadamard], call_graph_example=None)
