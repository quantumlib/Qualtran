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
from typing import Dict, Optional, Tuple, TYPE_CHECKING, Union

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
from qualtran._infra.composite_bloq import CompositeBloq
from qualtran.bloqs.arithmetic.addition import Add
from qualtran.bloqs.arithmetic.bitwise import BitwiseNot
from qualtran.bloqs.basic_gates import OnEach, XGate
from qualtran.bloqs.bookkeeping import Allocate, Cast, Free
from qualtran.bloqs.mcmt.multi_target_cnot import MultiTargetCNOT
from qualtran.drawing import Text

if TYPE_CHECKING:
    from qualtran.drawing import WireSymbol
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
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
        N = 2**b_bitsize
        if unsigned:
            return {'a': a, 'b': int((a - b) % N)}
        # Subtraction of signed integers can result in overflow. In most classical programming languages (e.g. C++)
        # what happens when an overflow happens is left as an implementation detail for compiler designers.
        # However for quantum subtraction the operation should be unitary and that means that the unitary of
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

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        delta = self.b_dtype.bitsize - self.a_dtype.bitsize
        costs = {
            OnEach(self.b_dtype.bitsize, XGate()): 3,
            Add(QUInt(self.b_dtype.bitsize), QUInt(self.b_dtype.bitsize)): 1,
        }
        if delta:
            costs[Allocate(QAny(delta))] = 1
            costs[Free(QAny(delta))] = 1
            if isinstance(self.a_dtype, QInt):
                costs[MultiTargetCNOT(delta)] = 2
        return costs

    def build_composite_bloq(self, bb: 'BloqBuilder', a: Soquet, b: Soquet) -> Dict[str, 'SoquetT']:
        delta = self.b_dtype.bitsize - self.a_dtype.bitsize
        n_bits = self.b_dtype.bitsize
        if delta:
            if not isinstance(delta, int):
                raise ValueError(
                    'Symbolic decomposition not supported when a_dtype != b_dtype'
                )  # pragma: no cover
            a_split = bb.split(a)
            prefix = bb.allocate(delta)
            if isinstance(self.a_dtype, QInt):
                a_split[0], prefix = bb.add(
                    MultiTargetCNOT(delta), control=a_split[0], targets=prefix
                )
            prefix_split = bb.split(prefix)
            a_split = np.concatenate([prefix_split, a_split])
            a = bb.join(a_split, QUInt(n_bits))
        else:
            a = bb.add(Cast(self.a_dtype, QUInt(n_bits)), reg=a)
        b = bb.add(Cast(self.b_dtype, QUInt(n_bits)), reg=b)
        a = bb.add(OnEach(n_bits, XGate()), q=a)
        a, b = bb.add(Add(QUInt(n_bits)), a=a, b=b)
        b = bb.add(OnEach(n_bits, XGate()), q=b)
        a = bb.add(OnEach(n_bits, XGate()), q=a)
        b = bb.add(Cast(QUInt(n_bits), self.b_dtype), reg=b)
        if delta:
            a_split = bb.split(a)
            prefix = bb.join(a_split[:delta])
            if isinstance(self.a_dtype, QInt):
                a_split[delta], prefix = bb.add(
                    MultiTargetCNOT(delta), control=a_split[delta], targets=prefix
                )
            prefix = bb.add(Cast(prefix.reg.dtype, QAny(delta)), reg=prefix)
            bb.free(prefix)
            a = bb.join(a_split[delta:], QUInt(self.a_dtype.bitsize))
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


@bloq_example
def _sub_symp_decomposition() -> CompositeBloq:
    n = sympy.Symbol('n')
    sub_symp_decomposition = Subtract(QInt(bitsize=n)).decompose_bloq()
    return sub_symp_decomposition


_SUB_DOC = BloqDocSpec(
    bloq_cls=Subtract,
    examples=[_sub_symb, _sub_small, _sub_large, _sub_diff_size_regs, _sub_symp_decomposition],
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

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {BitwiseNot(self.dtype): 2, Add(self.dtype, self.dtype): 1}

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
