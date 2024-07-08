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

from typing import Dict, Optional, Set, Tuple, TYPE_CHECKING, Union

import numpy as np
import sympy
from attrs import field, frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    QAny,
    QInt,
    QMontgomeryUInt,
    QUInt,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.arithmetic.addition import Add
from qualtran.bloqs.basic_gates import CNOT, OnEach, XGate
from qualtran.bloqs.bookkeeping import Allocate, Cast, Free
from qualtran.drawing import Text

if TYPE_CHECKING:
    from qualtran.drawing import WireSymbol
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class Subtract(Bloq):
    r"""An n-bit subtraction gate.

    Implements $U|a\rangle|b\rangle \rightarrow |a\rangle|a-b\rangle$ using $4n - 4$ T gates.

    This construction uses the relation `a - b = ~(~a + b)` to turn the operation into addition. This relation is used in
    [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization](https://arxiv.org/pdf/2007.07391)
    to turn addition into subtraction conditioned on a qubit.


    Args:
        a_dtype: Quantum datatype used to represent the integer a.
        b_dtype: Quantum datatype used to represent the integer b. Must be large
            enough to hold the result in the output register of a - b, or else it simply
            drops the most significant bits. If not specified, b_dtype is set to a_dtype.

    Registers:
        a: A a_dtype.bitsize-sized input register (register a above).
        b: A b_dtype.bitsize-sized input/output register (register b above).

    References:
        [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization, page 9](https://arxiv.org/pdf/2007.07391)
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
        N = 2**b_bitsize
        if unsigned:
            return {'a': a, 'b': int((a - b) % N)}
        # Subtraction of signed integers can result in overflow. In most classical programming languages (e.g. C++)
        # what happens when an overflow happens is left as an implementation detail for compiler designers.
        # However for quantum subtraction the operation shoudl be unitary and that means that the unitary of
        # the bloq should be a permutation matrix.
        # If we hold `a` constant then the valid range of values of `b` [-N/2, N/2) gets shifted forward or backwards
        # by `a`. to keep the operation unitary overflowing values wrap around. this is the same as moving the range [0, N)
        # by the same amount modulu $N$. that is add N/2 before subtraction and then remove it.
        half_n = N >> 1
        return {'a': a, 'b': int((a - b + half_n) % N) - half_n}

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
        delta = self.b_dtype.bitsize - self.a_dtype.bitsize
        return (
            {
                (OnEach(self.b_dtype.bitsize, XGate()), 3),
                (Add(QUInt(self.b_dtype.bitsize), QUInt(self.b_dtype.bitsize)), 1),
            }
            .union([(CNOT(), 2 * delta)] if isinstance(self.a_dtype, QInt) else [])
            .union([(Allocate(QAny(delta)), 1), (Free(QAny(delta)), 1)] if delta else [])
        )

    def build_composite_bloq(self, bb: 'BloqBuilder', a: Soquet, b: Soquet) -> Dict[str, 'SoquetT']:
        delta = self.b_dtype.bitsize - self.a_dtype.bitsize
        n_bits = self.b_dtype.bitsize
        a = bb.split(a)
        b = bb.split(b)
        if delta:
            prefix = bb.split(bb.allocate(delta))
            if isinstance(self.a_dtype, QInt):
                for i in range(delta):
                    a[0], prefix[i] = bb.add(CNOT(), ctrl=a[0], target=prefix[i])
            a = np.concatenate([prefix, a])
        a = bb.join(a, QUInt(n_bits))
        b = bb.join(b, QUInt(n_bits))
        a = bb.add(OnEach(n_bits, XGate()), q=a)
        a, b = bb.add(Add(QUInt(n_bits)), a=a, b=b)
        b = bb.add(OnEach(n_bits, XGate()), q=b)
        a = bb.add(OnEach(n_bits, XGate()), q=a)
        b = bb.add(Cast(QUInt(n_bits), self.b_dtype), reg=b)
        a = bb.split(a)
        if delta:
            if isinstance(self.a_dtype, QInt):
                for i in range(delta):
                    a[delta], a[i] = bb.add(CNOT(), ctrl=a[delta], target=a[i])
            bb.free(bb.join(a[:delta]))
        a = bb.join(a[delta:], QUInt(self.a_dtype.bitsize))
        a = bb.add(Cast(QUInt(self.a_dtype.bitsize), self.a_dtype), reg=a)
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
