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
from typing import Dict

import numpy as np
import sympy
from attrs import frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    BoundedQUInt,
    QBit,
    Register,
    Side,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.arithmetic import GreaterThan
from qualtran.bloqs.basic_gates import CSwap
from qualtran.symbolics import bit_length, SymbolicInt


@frozen
class Comparator(Bloq):
    r"""Compare and potentially swaps two numbers in {0, ... L - 1}.

    Implements $U|a\rangle|b\rangle|0\rangle \rightarrow |\min(a,b)\rangle|\max(a,b)\rangle|a>b\rangle$,

    where $a$ and $b$ are n-qubit quantum registers. On output a and b are
    swapped if a > b. Forms the base primitive for sorting.
    The numbers are represented using $n = \ceil{\log L}$ bits.

    Args:
        L: Upper-bound (excluded) on the input numbers.

    Registers:
        a: A n-bit-sized input register (register a above).
        b: A n-bit-sized input register (register b above).
        out: A single bit output register which will store the result of the comparator.

    References:
        [Improved techniques for preparing eigenstates of fermionic Hamiltonians](https://www.nature.com/articles/s41534-018-0071-5).
        Fig. 1. in main text.
    """

    L: SymbolicInt

    @property
    def signature(self):
        input_dtype = BoundedQUInt(self.bitsize, self.L)
        return Signature(
            [
                Register('a', input_dtype),
                Register('b', input_dtype),
                Register('out', QBit(), side=Side.RIGHT),
            ]
        )

    @property
    def bitsize(self):
        """number of bits to represent the inputs"""
        return bit_length(self.L - 1)

    def pretty_name(self) -> str:
        return "Cmprtr"

    def build_composite_bloq(
        self, bb: 'BloqBuilder', a: 'Soquet', b: 'Soquet'
    ) -> Dict[str, 'SoquetT']:
        out = bb.allocate(dtype=QBit())
        a, b, out = bb.add(GreaterThan(self.bitsize, self.bitsize), a=a, b=b, target=out)
        out, a, b = bb.add(CSwap(self.bitsize), ctrl=out, x=a, y=b)
        return {'a': a, 'b': b, 'out': out}


@bloq_example
def _cmp_symb() -> Comparator:
    L = sympy.Symbol('L')
    cmp_symb = Comparator(L=L)
    return cmp_symb


_COMPARATOR_DOC = BloqDocSpec(bloq_cls=Comparator, examples=[_cmp_symb])


@frozen
class BitonicSort(Bloq):
    r"""Sort k numbers in the range {0, ..., L-1}.

    TODO: actually implement the algorithm using comparitor. Hiding ancilla cost
        for the moment. Issue #219

    Args:
        L: Upper-bound (excluded) on the input integers.
        k: Number of integers to sort.

    Registers:
        input: List of k integers we want to sort.

    References:
        [Improved techniques for preparing eigenstates of fermionic Hamiltonians](https://www.nature.com/articles/s41534-018-0071-5).
        Supporting Information Sec. II.
    """

    L: SymbolicInt
    k: SymbolicInt

    @property
    def signature(self):
        return Signature([Register("input", BoundedQUInt(self.bitsize, self.L), shape=(self.k,))])

    @property
    def bitsize(self) -> SymbolicInt:
        return bit_length(self.L - 1)

    def pretty_name(self) -> str:
        return "BSort"

    def _t_complexity_(self):
        # Need O(k * log^2(k)) comparisons.
        # TODO: This is Big-O complexity.
        # Should work out constant factors or
        # See: https://github.com/quantumlib/Qualtran/issues/219
        return (
            self.k
            * int(np.ceil(max(np.log2(self.k) ** 2.0, 1)))
            * Comparator(self.bitsize).t_complexity()
        )


@bloq_example
def _bitonic_sort() -> BitonicSort:
    L = sympy.Symbol('L')
    bitonic_sort = BitonicSort(L=2**L + 1, k=3)
    return bitonic_sort


_BITONIC_SORT_DOC = BloqDocSpec(bloq_cls=BitonicSort, examples=[_bitonic_sort])
