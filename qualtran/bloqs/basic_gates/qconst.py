#  Copyright 2026 Google LLC
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
from typing import cast, Dict, Mapping, Optional, Set, Tuple, TYPE_CHECKING, Union

import numpy as np
import sympy
from attrs import frozen
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    DecomposeTypeError,
    QDType,
    QInt,
    QUInt,
    Register,
    Side,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.bookkeeping import ArbitraryClifford
from qualtran.drawing import directional_text_box, Text, WireSymbol
from qualtran.symbolics import is_symbolic, SymbolicInt

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, BloqCountT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValRetT, ClassicalValT


@frozen
class _QConst(Bloq):
    """Represent a classical integer vector (state or effect) with a given quantum data type.

    Args:
        val: the classical value.
        qdtype: The quantum data type.
        state: True if this is a state; an effect otherwise.

    Registers:
        val: The register of type `qdtype` which initializes/de-allocates the value `val`.
    """

    val: SymbolicInt
    qdtype: QDType
    state: bool

    def __attrs_post_init__(self):
        if is_symbolic(self.val) or is_symbolic(self.qdtype):
            return

        self.qdtype.assert_valid_classical_val(self.val)

    @cached_property
    def signature(self) -> Signature:
        side = Side.RIGHT if self.state else Side.LEFT
        return Signature([Register('val', self.qdtype, side=side)])

    @staticmethod
    def _build_composite_state(bb: 'BloqBuilder', bits: NDArray[np.uint8]) -> Dict[str, 'SoquetT']:
        from qualtran.bloqs.basic_gates.z_basis import OneState, ZeroState

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
        from qualtran.bloqs.basic_gates.z_basis import OneEffect, ZeroEffect

        xs = bb.split(val)
        effects = [ZeroEffect(), OneEffect()]
        for i, bit in enumerate(bits):
            bb.add(effects[bit], q=xs[i])
        return {}

    def build_composite_bloq(self, bb: 'BloqBuilder', **val: 'SoquetT') -> Dict[str, 'SoquetT']:
        if is_symbolic(self.qdtype) or is_symbolic(self.val):
            raise DecomposeTypeError(f'Symbolic qdtype {self.qdtype} or val {self.val} not supported')
        bits = np.asarray(self.qdtype.to_bits(self.val))
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
        return {ArbitraryClifford(self.qdtype.num_bits): 1}

    def adjoint(self) -> 'Bloq':
        return _QConst(val=self.val, qdtype=self.qdtype, state=not self.state)

    def __str__(self) -> str:
        s = f'{self.val}'
        return f'QConst({s})'

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('')

        return directional_text_box(text=f'{self.val}', side=reg.side)


@frozen
class QIntState(Bloq):
    """The state |val> for a signed integer val of type QInt(bitsize)

    Args:
        val: the classical value
        bitsize: The bitsize of the register

    Registers:
        val: The register of size `bitsize` which initializes the value `val`.
    """

    val: SymbolicInt
    bitsize: SymbolicInt

    def __attrs_post_init__(self):
        if is_symbolic(self.val) or is_symbolic(self.bitsize):
            return
        QInt(self.bitsize).assert_valid_classical_val(self.val)

    @cached_property
    def _impl(self):
        return _QConst(val=self.val, qdtype=QInt(self.bitsize), state=True)

    @property
    def signature(self) -> 'Signature':
        return self._impl.signature

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        return self._impl.build_composite_bloq(bb, **soqs)

    def on_classical_vals(
        self, **vals: Union['sympy.Symbol', 'ClassicalValT']
    ) -> Mapping[str, 'ClassicalValRetT']:
        return self._impl.on_classical_vals(**vals)

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        return self._impl.build_call_graph(ssa)

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        return self._impl.wire_symbol(reg, idx)

    def adjoint(self) -> 'QIntEffect':
        return QIntEffect(val=self.val, bitsize=self.bitsize)

    def __str__(self):
        return f'QIntState({self.val})'


@bloq_example
def _qint_state() -> QIntState:
    qint_state = QIntState(-5, bitsize=8)
    return qint_state


_QINT_STATE_DOC = BloqDocSpec(bloq_cls=QIntState, examples=[_qint_state])


@frozen
class QIntEffect(Bloq):
    """The effect <val| for a signed integer val of type QInt(bitsize)

    Args:
        val: the classical value
        bitsize: The bitsize of the register

    Registers:
        val: The register of size `bitsize` which de-allocates the value `val`.
    """

    val: SymbolicInt
    bitsize: SymbolicInt

    def __attrs_post_init__(self):
        if is_symbolic(self.val) or is_symbolic(self.bitsize):
            return
        QInt(self.bitsize).assert_valid_classical_val(self.val)

    @cached_property
    def _impl(self):
        return _QConst(val=self.val, qdtype=QInt(self.bitsize), state=False)

    @property
    def signature(self) -> 'Signature':
        return self._impl.signature

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        return self._impl.build_composite_bloq(bb, **soqs)

    def on_classical_vals(
        self, **vals: Union['sympy.Symbol', 'ClassicalValT']
    ) -> Mapping[str, 'ClassicalValRetT']:
        return self._impl.on_classical_vals(**vals)

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        return self._impl.build_call_graph(ssa)

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        return self._impl.wire_symbol(reg, idx)

    def adjoint(self) -> 'QIntState':
        return QIntState(val=self.val, bitsize=self.bitsize)

    def __str__(self):
        return f'QIntEffect({self.val})'


@bloq_example
def _qint_effect() -> QIntEffect:
    qint_effect = QIntEffect(-5, bitsize=8)
    return qint_effect


_QINT_EFFECT_DOC = BloqDocSpec(bloq_cls=QIntEffect, examples=[_qint_effect])


@frozen
class QUIntState(Bloq):
    """The state |val> for an unsigned integer val of type QUInt(bitsize)

    Args:
        val: the classical value
        bitsize: The bitsize of the register

    Registers:
        val: The register of size `bitsize` which initializes the value `val`.
    """

    val: SymbolicInt
    bitsize: SymbolicInt

    def __attrs_post_init__(self):
        if is_symbolic(self.val) or is_symbolic(self.bitsize):
            return
        QUInt(self.bitsize).assert_valid_classical_val(self.val)

    @cached_property
    def _impl(self):
        return _QConst(val=self.val, qdtype=QUInt(self.bitsize), state=True)

    @property
    def signature(self) -> 'Signature':
        return self._impl.signature

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        return self._impl.build_composite_bloq(bb, **soqs)

    def on_classical_vals(
        self, **vals: Union['sympy.Symbol', 'ClassicalValT']
    ) -> Mapping[str, 'ClassicalValRetT']:
        return self._impl.on_classical_vals(**vals)

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        return self._impl.build_call_graph(ssa)

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        return self._impl.wire_symbol(reg, idx)

    def adjoint(self) -> 'QUIntEffect':
        return QUIntEffect(val=self.val, bitsize=self.bitsize)

    def __str__(self):
        return f'QUIntState({self.val})'


@bloq_example
def _quint_state() -> QUIntState:
    quint_state = QUIntState(5, bitsize=8)
    return quint_state


_QUINT_STATE_DOC = BloqDocSpec(bloq_cls=QUIntState, examples=[_quint_state])


@frozen
class QUIntEffect(Bloq):
    """The effect <val| for an unsigned integer val of type QUInt(bitsize)

    Args:
        val: the classical value
        bitsize: The bitsize of the register

    Registers:
        val: The register of size `bitsize` which de-allocates the value `val`.
    """

    val: SymbolicInt
    bitsize: SymbolicInt

    def __attrs_post_init__(self):
        if is_symbolic(self.val) or is_symbolic(self.bitsize):
            return
        QUInt(self.bitsize).assert_valid_classical_val(self.val)

    @cached_property
    def _impl(self):
        return _QConst(val=self.val, qdtype=QUInt(self.bitsize), state=False)

    @property
    def signature(self) -> 'Signature':
        return self._impl.signature

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> Dict[str, 'SoquetT']:
        return self._impl.build_composite_bloq(bb, **soqs)

    def on_classical_vals(
        self, **vals: Union['sympy.Symbol', 'ClassicalValT']
    ) -> Mapping[str, 'ClassicalValRetT']:
        return self._impl.on_classical_vals(**vals)

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        return self._impl.build_call_graph(ssa)

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        return self._impl.wire_symbol(reg, idx)

    def adjoint(self) -> 'QUIntState':
        return QUIntState(val=self.val, bitsize=self.bitsize)

    def __str__(self):
        return f'QUIntEffect({self.val})'


@bloq_example
def _quint_effect() -> QUIntEffect:
    quint_effect = QUIntEffect(5, bitsize=8)
    return quint_effect


_QUINT_EFFECT_DOC = BloqDocSpec(bloq_cls=QUIntEffect, examples=[_quint_effect])
