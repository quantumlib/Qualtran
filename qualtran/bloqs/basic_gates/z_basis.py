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
from typing import cast, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import attrs
import numpy as np
import sympy
from attrs import frozen
from numpy.typing import NDArray

from qualtran import (
    AddControlledT,
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    CompositeBloq,
    ConnectionT,
    CtrlSpec,
    DecomposeTypeError,
    QAny,
    QBit,
    QDType,
    Register,
    Side,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.bookkeeping import ArbitraryClifford
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.drawing import Circle, directional_text_box, Text, TextBox, WireSymbol

if TYPE_CHECKING:
    import cirq
    import quimb.tensor as qtn

    from qualtran.cirq_interop import CirqQuregT
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator

_ZERO = np.array([1, 0], dtype=np.complex128)
_ONE = np.array([0, 1], dtype=np.complex128)
_PAULIZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)


@frozen
class _ZVector(Bloq):
    """The |0> or |1> state or effect.

    Please use the explicitly named subclasses instead of the boolean arguments.

    Args:
        bit: False chooses |0>, True chooses |1>
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

    def decompose_bloq(self) -> CompositeBloq:
        raise DecomposeTypeError(f"{self} is atomic")

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        import quimb.tensor as qtn

        side = outgoing if self.state else incoming
        return [
            qtn.Tensor(data=_ONE if self.bit else _ZERO, inds=[(side['q'], 0)], tags=[str(self)])
        ]

    def on_classical_vals(self, *, q: Optional[int] = None) -> Dict[str, int]:
        """Return or consume 1 or 0 depending on `self.state` and `self.bit`.

        If `self.state`, we return a bit in the `q` register. Otherwise,
        we assert that the inputted `q` register is the correct bit.
        """
        bit_int = 1 if self.bit else 0  # guard against bad `self.bit` types.
        if self.state:
            assert q is None
            return {'q': bit_int}

        assert q == bit_int, q
        return {}

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', **cirq_quregs: 'CirqQuregT'  # type: ignore[type-var]
    ) -> Tuple[Union['cirq.Operation', None], Dict[str, 'CirqQuregT']]:  # type: ignore[type-var]
        if not self.state:
            raise ValueError(f"There is no Cirq equivalent for {self}")

        import cirq

        (q,) = qubit_manager.qalloc(self.n)

        if self.bit:
            op = cirq.X(q)
        else:
            op = None
        return op, {'q': np.array([q])}

    def __str__(self) -> str:
        s = '1' if self.bit else '0'
        return f'|{s}>' if self.state else f'<{s}|'

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text('')
        s = '1' if self.bit else '0'
        return directional_text_box(s, side=reg.side)


def _hide_base_fields(cls, fields):
    # for use in attrs `field_transformer`.
    return [
        field.evolve(repr=False) if field.name in ['bit', 'state'] else field for field in fields
    ]


@frozen(init=False, field_transformer=_hide_base_fields)
class ZeroState(_ZVector):
    """The state |0>"""

    def __init__(self, n: int = 1):
        self.__attrs_init__(bit=False, state=True, n=n)

    def adjoint(self) -> 'Bloq':
        return ZeroEffect()


@bloq_example
def _zero_state() -> ZeroState:
    zero_state = ZeroState()
    return zero_state


_ZERO_STATE_DOC = BloqDocSpec(bloq_cls=ZeroState, examples=[_zero_state])


@frozen(init=False, field_transformer=_hide_base_fields)
class ZeroEffect(_ZVector):
    """The effect <0|"""

    def __init__(self, n: int = 1):
        self.__attrs_init__(bit=False, state=False, n=n)

    def adjoint(self) -> 'Bloq':
        return ZeroState()


@bloq_example
def _zero_effect() -> ZeroEffect:
    zero_effect = ZeroEffect()
    return zero_effect


_ZERO_EFFECT_DOC = BloqDocSpec(bloq_cls=ZeroEffect, examples=[_zero_effect])


@frozen(init=False, field_transformer=_hide_base_fields)
class OneState(_ZVector):
    """The state |1>"""

    def __init__(self, n: int = 1):
        self.__attrs_init__(bit=True, state=True, n=n)

    def adjoint(self) -> 'Bloq':
        return OneEffect()


@bloq_example
def _one_state() -> OneState:
    one_state = OneState()
    return one_state


_ONE_STATE_DOC = BloqDocSpec(bloq_cls=OneState, examples=[_one_state])


@frozen(init=False, field_transformer=_hide_base_fields)
class OneEffect(_ZVector):
    """The effect <1|"""

    def __init__(self, n: int = 1):
        self.__attrs_init__(bit=True, state=False, n=n)

    def adjoint(self) -> 'Bloq':
        return OneState()


@bloq_example
def _one_effect() -> OneEffect:
    one_effect = OneEffect()
    return one_effect


_ONE_EFFECT_DOC = BloqDocSpec(bloq_cls=OneEffect, examples=[_one_effect])


@frozen
class ZGate(Bloq):
    """The Z gate.

    This causes a phase flip: Z|+> = |-> and vice-versa.
    """

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(q=1)

    def adjoint(self) -> 'Bloq':
        return self

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic")

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        import quimb.tensor as qtn

        return [
            qtn.Tensor(
                data=_PAULIZ, inds=[(outgoing['q'], 0), (incoming['q'], 0)], tags=[str(self)]
            )
        ]

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> Tuple['Bloq', 'AddControlledT']:
        if ctrl_spec != CtrlSpec():
            # Delegate to the general superclass behavior
            return super().get_ctrl_system(ctrl_spec=ctrl_spec)

        bloq = CZ()

        def add_controlled(
            bb: 'BloqBuilder', ctrl_soqs: Sequence['SoquetT'], in_soqs: Dict[str, 'SoquetT']
        ) -> Tuple[Iterable['SoquetT'], Iterable['SoquetT']]:
            (ctrl_soq,) = ctrl_soqs
            ctrl_soq, q2 = bb.add(bloq, q1=ctrl_soq, q2=in_soqs['q'])
            return (ctrl_soq,), (q2,)

        return bloq, add_controlled

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', q: 'CirqQuregT'
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        import cirq

        (q,) = q
        return cirq.Z(q), {'q': np.asarray([q])}

    def _t_complexity_(self) -> 'TComplexity':
        return TComplexity(clifford=1)

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text('')

        return TextBox('Z')


@bloq_example
def _zgate() -> ZGate:
    zgate = ZGate()
    return zgate


_Z_GATE_DOC = BloqDocSpec(bloq_cls=ZGate, examples=[_zgate], call_graph_example=None)


@frozen
class CZ(Bloq):
    """Two-qubit controlled-Z gate.

    Registers:
        ctrl: One-bit control register.
        target: One-bit target register.
    """

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(q1=1, q2=1)

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic")

    def adjoint(self) -> 'Bloq':
        return self

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        import quimb.tensor as qtn

        unitary = np.diag(np.array([1, 1, 1, -1], dtype=np.complex128)).reshape((2, 2, 2, 2))
        inds = [(outgoing['q1'], 0), (outgoing['q2'], 0), (incoming['q1'], 0), (incoming['q2'], 0)]

        return [qtn.Tensor(data=unitary, inds=inds, tags=[str(self)])]

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', q1: 'CirqQuregT', q2: 'CirqQuregT'
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        import cirq

        (q1,) = q1
        (q2,) = q2
        return cirq.CZ(q1, q2), {'q1': np.array([q1]), 'q2': np.array([q2])}

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')
        if reg.name == 'q1' or reg.name == 'q2':
            return Circle()
        raise ValueError(f'Unknown wire symbol register name: {reg.name}')


@bloq_example
def _cz() -> CZ:
    cz = CZ()
    return cz


_CZ_DOC = BloqDocSpec(bloq_cls=CZ, examples=[_cz], call_graph_example=None)


@frozen
class _IntVector(Bloq):
    """Represent a classical non-negative integer vector (state or effect).

    Args:
        val: the classical value
        bitsize: The bitsize of the register
        state: True if this is a state; an effect otherwise.

    Registers:
        val: The register of size `bitsize` which initializes the value `val`.
    """

    val: Union[int, sympy.Expr] = attrs.field()
    bitsize: Union[int, sympy.Expr]
    state: bool

    @val.validator
    def check(self, attribute, val):
        if isinstance(val, sympy.Expr) or isinstance(self.bitsize, sympy.Expr):
            return

        if val < 0:
            raise ValueError("`val` must be positive")

        if val >= 2**self.bitsize:
            raise ValueError(f"`val` is too big for bitsize {self.bitsize}")

    @cached_property
    def dtype(self) -> QDType:
        if self.bitsize == 1:
            return QBit()
        return QAny(self.bitsize)

    @cached_property
    def signature(self) -> Signature:
        side = Side.RIGHT if self.state else Side.LEFT
        return Signature([Register('val', self.dtype, side=side)])

    @staticmethod
    def _build_composite_state(bb: 'BloqBuilder', bits: NDArray[np.uint8]) -> Dict[str, 'SoquetT']:
        states = [ZeroState(), OneState()]
        xs = []
        for bit in bits:
            x = bb.add(states[bit])
            xs.append(x)
        xs = np.array(xs)

        return {'val': bb.join(xs)}

    @staticmethod
    def _build_composite_effect(
        bb: 'BloqBuilder', val: 'Soquet', bits: NDArray[np.uint8]
    ) -> Dict[str, 'SoquetT']:
        xs = bb.split(val)
        effects = [ZeroEffect(), OneEffect()]
        for i, bit in enumerate(bits):
            bb.add(effects[bit], q=xs[i])
        return {}

    def build_composite_bloq(self, bb: 'BloqBuilder', **val: 'SoquetT') -> Dict[str, 'SoquetT']:
        if isinstance(self.bitsize, sympy.Expr):
            raise DecomposeTypeError(f'Symbolic bitsize {self.bitsize} not supported')
        bits = np.asarray(self.dtype.to_bits(self.val))
        if self.state:
            assert not val
            return self._build_composite_state(bb, bits)
        else:
            return self._build_composite_effect(bb, cast(Soquet, val['val']), bits)

    def on_classical_vals(self, *, val: Optional[int] = None) -> Dict[str, Union[int, sympy.Expr]]:
        if self.state:
            assert val is None
            return {'val': self.val}

        assert val == self.val, val
        return {}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {ArbitraryClifford(self.bitsize): 1}

    def __str__(self) -> str:
        s = f'{self.val}'
        return f'|{s}>' if self.state else f'<{s}|'

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')

        return directional_text_box(text=f'{self.val}', side=reg.side)


@frozen(init=False, field_transformer=_hide_base_fields)
class IntState(_IntVector):
    """The state |val> for non-negative integer val

    Args:
        val: the classical value
        bitsize: The bitsize of the register

    Registers:
        val: The register of size `bitsize` which initializes the value `val`.
    """

    def __init__(self, val: Union[int, sympy.Expr], bitsize: Union[int, sympy.Expr]):
        self.__attrs_init__(val=val, bitsize=bitsize, state=True)


@bloq_example
def _int_state() -> IntState:
    int_state = IntState(55, bitsize=8)
    return int_state


_INT_STATE_DOC = BloqDocSpec(bloq_cls=IntState, examples=[_int_state])


@frozen(init=False, field_transformer=_hide_base_fields)
class IntEffect(_IntVector):
    """The effect <val| for non-negative integer val

    Args:
        val: the classical value
        bitsize: The bitsize of the register

    Registers:
        val: The register of size `bitsize` which de-allocates the value `val`.
    """

    def __init__(self, val: int, bitsize: int):
        self.__attrs_init__(val=val, bitsize=bitsize, state=False)


@bloq_example
def _int_effect() -> IntEffect:
    int_effect = IntEffect(55, bitsize=8)
    return int_effect


_INT_EFFECT_DOC = BloqDocSpec(bloq_cls=IntEffect, examples=[_int_effect])
