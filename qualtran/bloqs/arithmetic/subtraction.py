#  Copyright 2024 Google LLC
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
import math
from typing import Dict, Optional, Set, Tuple, TYPE_CHECKING, Union

import sympy
from attrs import field, frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    QInt,
    QMontgomeryUInt,
    QUInt,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.arithmetic.addition import Add
from qualtran.bloqs.arithmetic.bitwise import BitwiseNot
from qualtran.bloqs.arithmetic.negate import Negate
from qualtran.drawing import Text

if TYPE_CHECKING:
    from qualtran.drawing import WireSymbol
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class Subtract(Bloq):
    r"""An n-bit subtraction gate.

    Implements $U|a\rangle|b\rangle \rightarrow |a\rangle|a-b\rangle$ using $4n - 4 T$ gates.

    This first negates $b$ (assuming a two's complement representation), and then adds $a$ into it.

    Args:
        a_dtype: Quantum datatype used to represent the integer a.
        b_dtype: Quantum datatype used to represent the integer b. Must be large
            enough to hold the result in the output register of a - b, or else it simply
            drops the most significant bits. If not specified, b_dtype is set to a_dtype.

    Registers:
        a: A a_dtype.bitsize-sized input register (register a above).
        b: A b_dtype.bitsize-sized input/output register (register b above).
    """

    a_dtype: Union[QInt, QUInt, QMontgomeryUInt] = field()
    b_dtype: Union[QInt, QUInt, QMontgomeryUInt] = field()

    @b_dtype.default
    def b_dtype_default(self):
        return self.a_dtype

    @a_dtype.validator
    def _a_dtype_validate(self, field, val):
        if not isinstance(val, (QInt, QUInt, QMontgomeryUInt)):
            raise ValueError("Only QInt, QUInt and QMontgomeryUInt types are supported.")
        if isinstance(val.num_qubits, sympy.Expr):
            return
        if val.bitsize > self.b_dtype.bitsize:
            raise ValueError("a_dtype bitsize must be less than or equal to b_dtype bitsize")

    @b_dtype.validator
    def _b_dtype_validate(self, field, val):
        if not isinstance(val, (QInt, QUInt, QMontgomeryUInt)):
            raise ValueError("Only QInt, QUInt and QMontgomeryUInt types are supported.")

    @property
    def dtype(self):
        if self.a_dtype != self.b_dtype:
            raise ValueError(
                "Subtract.dtype is only supported when both operands have the same dtype: "
                f"{self.a_dtype=}, {self.b_dtype=}"
            )
        return self.a_dtype

    @property
    def signature(self):
        return Signature([Register("a", self.a_dtype), Register("b", self.b_dtype)])

    def _dtype_as_unsigned(
        self, dtype: Union[QInt, QUInt, QMontgomeryUInt]
    ) -> Union[QUInt, QMontgomeryUInt]:
        return dtype if not isinstance(dtype, QInt) else QUInt(dtype.bitsize)

    @property
    def a_dtype_as_unsigned(self):
        return self._dtype_as_unsigned(self.a_dtype)

    @property
    def b_dtype_as_unsigned(self):
        return self._dtype_as_unsigned(self.b_dtype)

    def on_classical_vals(
        self, a: 'ClassicalValT', b: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        unsigned = isinstance(self.a_dtype, (QUInt, QMontgomeryUInt))
        b_bitsize = self.b_dtype.bitsize
        N = 2**b_bitsize if unsigned else 2 ** (b_bitsize - 1)
        return {'a': a, 'b': int(math.fmod(a - b, N))}

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        from qualtran.drawing import directional_text_box

        if reg is None:
            return Text('')
        if reg.name == 'a':
            return directional_text_box('a', side=reg.side)
        elif reg.name == 'b':
            return directional_text_box('a-b', side=reg.side)
        else:
            raise ValueError()

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {
            (Negate(self.b_dtype), 1),
            (Add(self.a_dtype_as_unsigned, self.b_dtype_as_unsigned), 1),
        }

    def build_composite_bloq(self, bb: 'BloqBuilder', a: Soquet, b: Soquet) -> Dict[str, 'SoquetT']:
        b = bb.add(Negate(self.b_dtype), x=b)
        a, b = bb.add(Add(self.a_dtype_as_unsigned, self.b_dtype_as_unsigned), a=a, b=b)  # a - b
        return {'a': a, 'b': b}


@bloq_example
def _sub_symb() -> Subtract:
    n = sympy.Symbol('n')
    sub_symb = Subtract(QInt(bitsize=n))
    return sub_symb


@bloq_example
def _sub_small() -> Subtract:
    sub_small = Subtract(QInt(bitsize=4))
    return sub_small


@bloq_example
def _sub_large() -> Subtract:
    sub_large = Subtract(QInt(bitsize=64))
    return sub_large


@bloq_example
def _sub_diff_size_regs() -> Subtract:
    sub_diff_size_regs = Subtract(QInt(bitsize=4), QInt(bitsize=16))
    return sub_diff_size_regs


_SUB_DOC = BloqDocSpec(
    bloq_cls=Subtract, examples=[_sub_symb, _sub_small, _sub_large, _sub_diff_size_regs]
)


@frozen
class SubtractFrom(Bloq):
    """A version of `Subtract` that subtracts the first register from the second in place.

    Implements $U|a\rangle|b\rangle \rightarrow |a\rangle|b - a\rangle$, essentially equivalent to
    the statement `b -= a`.

    Args:
        dtype: Quantum datatype used to represent the integers a, b, and b - a.

    Registers:
        a: A dtype.bitsize-sized input register (register a above).
        b: A dtype.bitsize-sized input/output register (register b above).
    """

    dtype: Union[QInt, QUInt, QMontgomeryUInt] = field()

    @dtype.validator
    def _dtype_validate(self, field, val):
        if not isinstance(val, (QInt, QUInt, QMontgomeryUInt)):
            raise ValueError("Only QInt, QUInt and QMontgomeryUInt types are supported.")

    @property
    def signature(self):
        return Signature([Register("a", self.dtype), Register("b", self.dtype)])

    def on_classical_vals(
        self, a: 'ClassicalValT', b: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        unsigned = isinstance(self.dtype, (QUInt, QMontgomeryUInt))
        bitsize = self.dtype.bitsize
        N = 2**bitsize if unsigned else 2 ** (bitsize - 1)
        return {'a': a, 'b': int(math.fmod(b - a, N))}

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        from qualtran.drawing import directional_text_box

        if reg is None:
            return Text('')
        if reg.name == 'a':
            return directional_text_box('a', side=reg.side)
        elif reg.name == 'b':
            return directional_text_box('b - a', side=reg.side)
        else:
            raise ValueError()

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {(BitwiseNot(self.dtype), 2), (Add(self.dtype, self.dtype), 1)}

    def build_composite_bloq(self, bb: 'BloqBuilder', a: Soquet, b: Soquet) -> Dict[str, 'SoquetT']:
        b = bb.add(BitwiseNot(self.dtype), x=b)  # a, -1 - b
        a, b = bb.add_t(Add(self.dtype, self.dtype), a=a, b=b)  # a, a - 1 - b
        b = bb.add(BitwiseNot(self.dtype), x=b)  # a, -1 - (a - 1 - b) = a, -a + b
        return {'a': a, 'b': b}


@bloq_example
def _sub_from_symb() -> SubtractFrom:
    n = sympy.Symbol('n')
    sub_from_symb = SubtractFrom(QInt(bitsize=n))
    return sub_from_symb


@bloq_example
def _sub_from_small() -> SubtractFrom:
    sub_from_small = SubtractFrom(QInt(bitsize=4))
    return sub_from_small


@bloq_example
def _sub_from_large() -> SubtractFrom:
    sub_from_large = SubtractFrom(QInt(bitsize=64))
    return sub_from_large


_SUB_FROM_DOC = BloqDocSpec(
    bloq_cls=SubtractFrom, examples=[_sub_from_symb, _sub_from_small, _sub_from_large]
)
