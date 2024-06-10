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

import numpy as np
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
from qualtran.bloqs.arithmetic.addition import Add, AddK
from qualtran.bloqs.basic_gates import XGate
from qualtran.drawing import Text

if TYPE_CHECKING:
    from qualtran.drawing import WireSymbol
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class Subtract(Bloq):
    r"""An n-bit subtraction gate.

    Implements $U|a\rangle|b\rangle \rightarrow |a\rangle|a-b\rangle$ using $4n - 4 T$ gates.

    This construction uses `XGate` and `AddK` to compute the twos-compliment of `b` before
    doing a standard `Add`.

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
            raise ValueError("Only QInt, QUInt and QMontgomerUInt types are supported.")
        if isinstance(val.num_qubits, sympy.Expr):
            return
        if val.bitsize > self.b_dtype.bitsize:
            raise ValueError("a_dtype bitsize must be less than or equal to b_dtype bitsize")

    @b_dtype.validator
    def _b_dtype_validate(self, field, val):
        if not isinstance(val, (QInt, QUInt, QMontgomeryUInt)):
            raise ValueError("Only QInt, QUInt and QMontgomerUInt types are supported.")

    @property
    def dtype(self):
        if self.a_dtype != self.b_dtype:
            raise ValueError(
                "Add.dtype is only supported when both operands have the same dtype: "
                f"{self.a_dtype=}, {self.b_dtype=}"
            )
        return self.a_dtype

    @property
    def signature(self):
        return Signature([Register("a", self.a_dtype), Register("b", self.b_dtype)])

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
        a_dtype = (
            self.a_dtype if not isinstance(self.a_dtype, QInt) else QUInt(self.a_dtype.bitsize)
        )
        b_dtype = (
            self.b_dtype if not isinstance(self.b_dtype, QInt) else QUInt(self.b_dtype.bitsize)
        )
        return {
            (XGate(), self.b_dtype.bitsize),
            (AddK(self.b_dtype.bitsize, k=1), 1),
            (Add(a_dtype, b_dtype), 1),
        }

    def build_composite_bloq(self, bb: 'BloqBuilder', a: Soquet, b: Soquet) -> Dict[str, 'SoquetT']:
        b = np.array([bb.add(XGate(), q=q) for q in bb.split(b)])  # 1s complement of b.
        b = bb.add(
            AddK(self.b_dtype.bitsize, k=1), x=bb.join(b, self.b_dtype)
        )  # 2s complement of b.

        a_dtype = (
            self.a_dtype if not isinstance(self.a_dtype, QInt) else QUInt(self.a_dtype.bitsize)
        )
        b_dtype = (
            self.b_dtype if not isinstance(self.b_dtype, QInt) else QUInt(self.b_dtype.bitsize)
        )

        a, b = bb.add(Add(a_dtype, b_dtype), a=a, b=b)  # a - b
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
