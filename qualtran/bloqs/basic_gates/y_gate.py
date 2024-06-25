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

_PAULIY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)


@frozen
class YGate(Bloq):
    """The Pauli Y gate.

    This causes a bit flip (with a complex phase): Y|0> = 1j|1>, Y|1> = -1j|0>

    Registers:
        q: The qubit.
    """

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(q=1)

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic.")

    def adjoint(self) -> 'Bloq':
        return self

    def add_my_tensors(
        self,
        tn: 'qtn.TensorNetwork',
        tag: Any,
        *,
        incoming: Dict[str, SoquetT],
        outgoing: Dict[str, SoquetT],
    ):
        import quimb.tensor as qtn

        tn.add(qtn.Tensor(data=_PAULIY, inds=(outgoing['q'], incoming['q']), tags=["Y", tag]))

    def get_ctrl_system(
        self, ctrl_spec: Optional['CtrlSpec'] = None
    ) -> Tuple['Bloq', 'AddControlledT']:
        if not (ctrl_spec is None or ctrl_spec == CtrlSpec()):
            return super().get_ctrl_system(ctrl_spec=ctrl_spec)

        bloq = CYGate()

        def _add_ctrled(
            bb: 'BloqBuilder', ctrl_soqs: Sequence['SoquetT'], in_soqs: Dict[str, 'SoquetT']
        ) -> Tuple[Iterable['SoquetT'], Iterable['SoquetT']]:
            (ctrl,) = ctrl_soqs
            ctrl, q = bb.add(bloq, ctrl=ctrl, q=in_soqs['q'])
            return ((ctrl,), (q,))

        return bloq, _add_ctrled

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', q: 'CirqQuregT'
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        import cirq

        (q,) = q
        return cirq.Y(q), {'q': np.asarray([q])}

    def _t_complexity_(self) -> 'TComplexity':
        return TComplexity(clifford=1)

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text('')
        return TextBox('Y')

    def __str__(self):
        return 'Y'


@bloq_example
def _y_gate() -> YGate:
    y_gate = YGate()
    return y_gate


_Y_GATE_DOC = BloqDocSpec(bloq_cls=YGate, examples=[_y_gate], call_graph_example=None)


@frozen
class CYGate(Bloq):
    """A controlled Y gate.

    Registers:
        ctrl: The control qubit.
        q: The target qubit.
    """

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(ctrl=1, q=1)

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic.")

    def adjoint(self) -> 'Bloq':
        return self

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text('')
        if reg.name == 'ctrl':
            return Circle()
        if reg.name == 'q':
            return TextBox('Y')
        raise ValueError(f"Unknown register {reg}.")

    def __str__(self):
        return 'CY'


@bloq_example
def _cy_gate() -> CYGate:
    cy_gate = YGate().controlled()
    return cy_gate


_CY_GATE_DOC = BloqDocSpec(bloq_cls=CYGate, examples=[_cy_gate], call_graph_example=None)
