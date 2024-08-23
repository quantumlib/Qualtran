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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import numpy as np
from attrs import frozen

from qualtran import (
    AddControlledT,
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    ConnectionT,
    CtrlSpec,
    QBit,
    Register,
    Side,
    Signature,
    SoquetT,
)
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.drawing import directional_text_box, Text, WireSymbol

if TYPE_CHECKING:
    import cirq
    import quimb.tensor as qtn

    from qualtran.cirq_interop import CirqQuregT
    from qualtran.simulation.classical_sim import ClassicalValT

_PLUS = np.ones(2, dtype=np.complex128) / np.sqrt(2)
_MINUS = np.array([1, -1], dtype=np.complex128) / np.sqrt(2)
_PAULIX = np.array([[0, 1], [1, 0]], dtype=np.complex128)


@frozen
class _XVector(Bloq):
    """The |+> or |-> state or effect.

    Please use the explicitly named subclasses instead of the boolean arguments.

    Args:
        bit: False chooses |+>, True chooses |->
        state: True means this is a state with right registers; False means this is an
            effect with left registers.
        n: bitsize of the vector.

    """

    bit: bool
    state: bool = True
    n: int = 1

    def __attrs_post_init__(self):
        if self.n != 1:
            raise NotImplementedError("Come back later.")

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('q', QBit(), side=Side.RIGHT if self.state else Side.LEFT)])

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        import quimb.tensor as qtn

        side = outgoing if self.state else incoming
        return [
            qtn.Tensor(data=_MINUS if self.bit else _PLUS, inds=[(side['q'], 0)], tags=[str(self)])
        ]

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', **cirq_quregs: 'CirqQuregT'  # type: ignore[type-var]
    ) -> Tuple[Union['cirq.Operation', None], Dict[str, 'CirqQuregT']]:  # type: ignore[type-var]
        if not self.state:
            raise ValueError(f"There is no Cirq equivalent for {self}")

        import cirq

        (q,) = qubit_manager.qalloc(self.n)

        if self.bit:
            op = cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(q), cirq.H(q)))
        else:
            op = cirq.H(q)

        return op, {'q': np.array([q])}

    def __str__(self) -> str:
        s = '-' if self.bit else '+'
        return f'|{s}>' if self.state else f'<{s}|'

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text('')
        s = '-' if self.bit else '+'
        return directional_text_box(s, side=reg.side)


def _hide_base_fields(cls, fields):
    # for use in attrs `field_trasnformer`.
    return [
        field.evolve(repr=False) if field.name in ['bit', 'state'] else field for field in fields
    ]


@frozen(init=False, field_transformer=_hide_base_fields)
class PlusState(_XVector):
    """The state |+>"""

    def __init__(self, n: int = 1):
        self.__attrs_init__(bit=False, state=True, n=n)

    def adjoint(self) -> 'Bloq':
        return PlusEffect()


@bloq_example
def _plus_state() -> PlusState:
    plus_state = PlusState()
    return plus_state


_PLUS_STATE_DOC = BloqDocSpec(bloq_cls=PlusState, examples=[_plus_state])


@frozen(init=False, field_transformer=_hide_base_fields)
class PlusEffect(_XVector):
    """The effect <+|"""

    def __init__(self, n: int = 1):
        self.__attrs_init__(bit=False, state=False, n=n)

    def adjoint(self) -> 'Bloq':
        return PlusState()


@bloq_example
def _plus_effect() -> PlusEffect:
    plus_effect = PlusEffect()
    return plus_effect


_PLUS_EFFECT_DOC = BloqDocSpec(bloq_cls=PlusEffect, examples=[_plus_effect])


@frozen(init=False, field_transformer=_hide_base_fields)
class MinusState(_XVector):
    """The state |->"""

    def __init__(self, n: int = 1):
        self.__attrs_init__(bit=True, state=True, n=n)

    def adjoint(self) -> 'Bloq':
        return MinusEffect()


@bloq_example
def _minus_state() -> MinusState:
    minus_state = MinusState()
    return minus_state


_MINUS_STATE_DOC = BloqDocSpec(bloq_cls=MinusState, examples=[_minus_state])


@frozen(init=False, field_transformer=_hide_base_fields)
class MinusEffect(_XVector):
    """The effect <-|"""

    def __init__(self, n: int = 1):
        self.__attrs_init__(bit=True, state=False, n=n)

    def adjoint(self) -> 'Bloq':
        return MinusState()


@bloq_example
def _minus_effect() -> MinusEffect:
    minus_effect = MinusEffect()
    return minus_effect


_MINUS_EFFECT_DOC = BloqDocSpec(bloq_cls=MinusEffect, examples=[_minus_effect])


@frozen
class XGate(Bloq):
    """The Pauli X gate.

    This causes a bit flip: X|0> = |1> and vice-versa.
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
                data=_PAULIX, inds=[(outgoing['q'], 0), (incoming['q'], 0)], tags=[str(self)]
            )
        ]

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> Tuple['Bloq', 'AddControlledT']:
        from qualtran.bloqs.basic_gates import CNOT, Toffoli

        if ctrl_spec == CtrlSpec():
            bloq: 'Bloq' = CNOT()
        elif ctrl_spec == CtrlSpec(cvs=(1, 1)):
            bloq = Toffoli()
        else:
            return super().get_ctrl_system(ctrl_spec)

        def add_controlled(
            bb: 'BloqBuilder', ctrl_soqs: Sequence['SoquetT'], in_soqs: Dict[str, 'SoquetT']
        ) -> Tuple[Iterable['SoquetT'], Iterable['SoquetT']]:
            (ctrl_soq,) = ctrl_soqs
            ctrl_soq, target = bb.add(bloq, ctrl=ctrl_soq, target=in_soqs['q'])
            return (ctrl_soq,), (target,)

        return bloq, add_controlled

    def on_classical_vals(self, q: int) -> Dict[str, 'ClassicalValT']:
        return {'q': (q + 1) % 2}

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', **cirq_quregs: 'CirqQuregT'
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        import cirq

        q = cirq_quregs.pop('q')

        (q,) = q
        return cirq.X(q), {'q': np.asarray([q])}

    def _t_complexity_(self):
        return TComplexity(clifford=1)

    def wire_symbol(self, reg: Register, idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        from qualtran.drawing import ModPlus

        if reg is None:
            return Text('X')

        return ModPlus()
