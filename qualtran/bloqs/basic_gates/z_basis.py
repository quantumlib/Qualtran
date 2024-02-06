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
from typing import Any, Dict, Optional, Set, Tuple, TYPE_CHECKING, Union

import attrs
import numpy as np
import quimb.tensor as qtn
import sympy
from attrs import frozen
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    CompositeBloq,
    DecomposeTypeError,
    QAny,
    QBit,
    Register,
    Side,
    Signature,
    SoquetT,
)
from qualtran.bloqs.util_bloqs import ArbitraryClifford
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.resource_counting import big_O
from qualtran.simulation.classical_sim import ints_to_bits

if TYPE_CHECKING:
    import cirq

    from qualtran.cirq_interop import CirqQuregT
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT

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

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        tag: Any,
        *,
        incoming: Dict[str, SoquetT],
        outgoing: Dict[str, SoquetT],
    ):
        side = outgoing if self.state else incoming
        tn.add(
            qtn.Tensor(
                data=_ONE if self.bit else _ZERO, inds=(side['q'],), tags=[self.short_name(), tag]
            )
        )

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
        self, qubit_manager: 'cirq.QubitManager', **cirq_quregs: 'CirqQuregT'
    ) -> Tuple[Union['cirq.Operation', None], Dict[str, 'CirqQuregT']]:
        if not self.state:
            raise ValueError(f"There is no Cirq equivalent for {self}")

        import cirq

        (q,) = qubit_manager.qalloc(self.n)

        if self.bit:
            op = cirq.X(q)
        else:
            op = None
        return op, {'q': np.array([q])}

    def pretty_name(self) -> str:
        s = self.short_name()
        return f'|{s}>' if self.state else f'<{s}|'

    def short_name(self) -> str:
        return '1' if self.bit else '0'


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

    def short_name(self) -> 'str':
        return 'Z'

    def decompose_bloq(self) -> CompositeBloq:
        raise DecomposeTypeError(f"{self} is atomic")

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
                data=_PAULIZ, inds=(outgoing['q'], incoming['q']), tags=[self.short_name(), tag]
            )
        )

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', q: 'CirqQuregT'
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        import cirq

        (q,) = q
        return cirq.Z(q), {'q': [q]}


@bloq_example
def _zgate() -> ZGate:
    zgate = ZGate()
    return zgate


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

    val: int = attrs.field()
    bitsize: int
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
    def signature(self) -> Signature:
        side = Side.RIGHT if self.state else Side.LEFT
        if self.bitsize == 1:
            return Signature([Register('val', QBit(), side=side)])
        return Signature([Register('val', QAny(self.bitsize), side=side)])

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
        bb: 'BloqBuilder', val: 'SoquetT', bits: NDArray[np.uint8]
    ) -> Dict[str, 'SoquetT']:
        xs = bb.split(val)
        effects = [ZeroEffect(), OneEffect()]
        for i, bit in enumerate(bits):
            bb.add(effects[bit], q=xs[i])
        return {}

    def build_composite_bloq(self, bb: 'BloqBuilder', **val: 'SoquetT') -> Dict[str, 'SoquetT']:
        bits = ints_to_bits(np.array([self.val]), w=self.bitsize)[0]
        if self.state:
            assert not val
            return self._build_composite_state(bb, bits)
        else:
            val = val['val']
            return self._build_composite_effect(bb, val, bits)

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        tag: Any,
        *,
        incoming: Dict[str, SoquetT],
        outgoing: Dict[str, SoquetT],
    ):
        data = np.zeros(2**self.bitsize).reshape((2,) * self.bitsize)
        bitstring = ints_to_bits(np.array([self.val]), w=self.bitsize)[0]
        data[tuple(bitstring)] = 1
        data = data.reshape(-1)

        if self.state:
            inds = (outgoing['val'],)
        else:
            inds = (incoming['val'],)

        tn.add(qtn.Tensor(data=data, inds=inds, tags=[self.short_name(), tag]))

    def on_classical_vals(self, *, val: Optional[int] = None) -> Dict[str, int]:
        if self.state:
            assert val is None
            return {'val': self.val}

        assert val == self.val, val

    def t_complexity(self) -> 'TComplexity':
        return TComplexity()

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {(ArbitraryClifford(self.bitsize), big_O(1))}

    def short_name(self) -> str:
        return f'{self.val}'

    def pretty_name(self) -> str:
        s = self.short_name()
        return f'|{s}>' if self.state else f'<{s}|'


@frozen(init=False, field_transformer=_hide_base_fields)
class IntState(_IntVector):
    """The state |val> for non-negative integer val

    Args:
        val: the classical value
        bitsize: The bitsize of the register

    Registers:
        val: The register of size `bitsize` which initializes the value `val`.
    """

    def __init__(self, val: int, bitsize: int):
        self.__attrs_init__(val=val, bitsize=bitsize, state=True)


@bloq_example
def _int_state() -> IntState:
    int_state = IntState(55, bitsize=8)
    return int_state


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
