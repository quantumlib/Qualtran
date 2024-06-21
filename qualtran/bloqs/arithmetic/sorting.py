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
from typing import Dict, Set

import numpy as np
import sympy
from attrs import frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    DecomposeTypeError,
    QBit,
    QUInt,
    Register,
    Side,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.arithmetic import GreaterThan
from qualtran.bloqs.basic_gates import CSwap
from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
from qualtran.symbolics import bit_length, is_symbolic, SymbolicInt


@frozen
class Comparator(Bloq):
    r"""Compare and potentially swaps two n-bit numbers.

    Implements $|a\rangle|b\rangle \mapsto |\min(a,b)\rangle|\max(a,b)\rangle|a>b\rangle$,

    where $a$ and $b$ are n-qubit quantum registers. On output a and b are
    swapped if a > b. Forms the base primitive for sorting.

    Args:
        bitsize: value of $n$ (i.e. the inputs are $n$-bit numbers).

    Registers:
        a: A n-bit-sized input register (register a above).
        b: A n-bit-sized input register (register b above).
        out (RIGHT): A single bit output register which will store the result of the comparator.

    References:
        [Improved techniques for preparing eigenstates of fermionic Hamiltonians](https://www.nature.com/articles/s41534-018-0071-5).
        Fig. 1. in main text.
    """

    bitsize: SymbolicInt

    @property
    def signature(self):
        input_dtype = QUInt(self.bitsize)
        return Signature(
            [
                Register('a', input_dtype),
                Register('b', input_dtype),
                Register('out', QBit(), side=Side.RIGHT),
            ]
        )

    def pretty_name(self) -> str:
        return "Cmprtr"

    def build_composite_bloq(
        self, bb: 'BloqBuilder', a: 'Soquet', b: 'Soquet'
    ) -> dict[str, 'SoquetT']:
        out = bb.allocate(dtype=QBit())
        a, b, out = bb.add(GreaterThan(self.bitsize, self.bitsize), a=a, b=b, target=out)
        out, a, b = bb.add(CSwap(self.bitsize), ctrl=out, x=a, y=b)
        return {'a': a, 'b': b, 'out': out}


@bloq_example
def _comparator() -> Comparator:
    comparator = Comparator(7)
    return comparator


@bloq_example
def _comparator_symb() -> Comparator:
    n = sympy.Symbol('n')
    comparator_symb = Comparator(n)
    return comparator_symb


_COMPARATOR_DOC = BloqDocSpec(bloq_cls=Comparator, examples=[_comparator, _comparator_symb])


@frozen
class ParallelComparators(Bloq):
    """Given k n-bit numbers, for each pair that is `offset` apart, compare and swap if needed to make them ordered.

    For each block of `2 * offset` numbers, apply a `Comparator` between each pair that is `offset` apart.
    Uses up to $k / 2$ Comparators.

    This is used by `BitonicMerge` to apply parallel merges with offsets 1, 2, 4 and so on.

    Args:
        k: size of the input list.
        offset: compare numbers whose indices are offset apart.
        bitsize: value of $n$ (i.e. the inputs are $n$-bit numbers).

    Registers:
        xs: input list of numbers.
        junk (RIGHT): ancilla generated by comparators.
    """

    k: SymbolicInt
    offset: SymbolicInt
    bitsize: SymbolicInt

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("xs", QUInt(self.bitsize), shape=(self.k,)),
                Register("junk", QBit(), shape=(self.num_comparisons,), side=Side.RIGHT),
            ]
        )

    def is_symbolic(self):
        return is_symbolic(self.bitsize, self.k, self.offset)

    @cached_property
    def num_comparisons(self) -> SymbolicInt:
        """Number of `Comparator` gates used in the decomposition"""
        if is_symbolic(self.k, self.offset):
            return self.k // 2  # upper bound

        full = (self.k // (2 * self.offset)) * self.offset
        rest = self.k % (2 * self.offset)
        return full + max(rest - self.offset, 0)

    def build_composite_bloq(self, bb: 'BloqBuilder', xs: 'SoquetT') -> Dict[str, 'SoquetT']:
        if self.is_symbolic():
            raise DecomposeTypeError(f"Cannot decompose symbolic {self=}")

        # make mypy happy
        assert not isinstance(self.k, sympy.Expr)
        assert not isinstance(self.offset, sympy.Expr)
        assert isinstance(xs, np.ndarray)

        comp = Comparator(self.bitsize)

        junk = []
        for start in range(0, self.k, 2 * self.offset):
            for step in range(self.offset):
                i, j = start + step, start + step + self.offset
                if j >= self.k:
                    break
                xs[i], xs[j], anc = bb.add(comp, a=xs[i], b=xs[j])
                junk.append(anc)

        assert len(junk) == self.num_comparisons

        return {'xs': xs, 'junk': np.array(junk)}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {(Comparator(self.bitsize), self.num_comparisons)}


@bloq_example
def _parallel_compare() -> ParallelComparators:
    parallel_compare = ParallelComparators(7, 2, bitsize=3)
    return parallel_compare


_PARALLEL_COMPARATORS_DOC = BloqDocSpec(
    bloq_cls=ParallelComparators,
    examples=[_parallel_compare],
    import_line="from qualtran.bloqs.arithmetic.sorting import ParallelComparators",
)


@frozen
class BitonicMerge(Bloq):
    r"""Merge two sorted sequences of n-bit integers.

    Given two sorted lists of length `half_length`, merges them inplace into a single
    sorted list.

    Currently only supports `half_length` equal to a power of two (#1090).

    If each half has length $k$, then the merge network uses $k * (\log{k} + 1)$ comparisons
    when $k$ is a power of 2.

    Args:
        half_length: Number of integers in each half
        bitsize: value of $n$ (i.e. the inputs are $n$-bit numbers).

    Registers:
        xs (LEFT): first input list of size `half_length`
        ys (LEFT): second input list of size `half_length`
        result (RIGHT): merged output list of size `2 * half_length`
        junk (RIGHT): ancilla generated by comparators.
    """
    half_length: SymbolicInt
    bitsize: SymbolicInt

    def __attrs_post_init__(self):
        k = self.half_length
        if not is_symbolic(k):
            assert not isinstance(k, sympy.Expr)
            assert k >= 1, "length of input lists must be positive"
            # TODO(#1090) support non-power-of-two input lengths
            assert (k & (k - 1)) == 0, "length of input lists must be a power of 2"

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register("xs", QUInt(self.bitsize), shape=(self.half_length,), side=Side.LEFT),
                Register("ys", QUInt(self.bitsize), shape=(self.half_length,), side=Side.LEFT),
                Register(
                    "result", QUInt(self.bitsize), shape=(self.half_length * 2,), side=Side.RIGHT
                ),
                Register("junk", QBit(), shape=(self.num_comparisons,), side=Side.RIGHT),
            ]
        )

    @cached_property
    def num_comparisons(self) -> SymbolicInt:
        """Number of `Comparator` gates used in the decomposition"""
        return self.half_length * (bit_length(self.half_length - 1) + 1)

    def is_symbolic(self):
        return is_symbolic(self.bitsize, self.half_length)

    def build_composite_bloq(
        self, bb: 'BloqBuilder', xs: 'SoquetT', ys: 'SoquetT'
    ) -> dict[str, 'SoquetT']:
        if self.is_symbolic():
            raise DecomposeTypeError(f"Cannot decompose symbolic {self=}")

        assert isinstance(xs, np.ndarray)
        assert isinstance(ys, np.ndarray)

        k = int(self.half_length)

        first_round_junk = []
        for i in range(k):
            xs[i], ys[k - 1 - i], anc = bb.add(Comparator(self.bitsize), a=xs[i], b=ys[k - 1 - i])
            first_round_junk.append(anc)

        result = np.concatenate([xs, ys])
        logk = int(bit_length(k - 1))
        assert 2**logk == k

        all_junks = [first_round_junk]
        for log_offset in reversed(range(logk)):
            result, ancs = bb.add(
                ParallelComparators(self.half_length * 2, 2**log_offset, self.bitsize), xs=result
            )
            all_junks.append(ancs)

        junk = np.concatenate(all_junks)
        assert len(junk) == self.num_comparisons

        return {'result': result, 'junk': np.array(junk)}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {(Comparator(self.bitsize), self.num_comparisons)}


@bloq_example
def _bitonic_merge() -> BitonicMerge:
    bitonic_merge = BitonicMerge(4, 7)
    return bitonic_merge


_BITONIC_MERGE_DOC = BloqDocSpec(
    bloq_cls=BitonicMerge,
    examples=[_bitonic_merge],
    import_line="from qualtran.bloqs.arithmetic.sorting import BitonicMerge",
)


@frozen
class BitonicSort(Bloq):
    r"""Sort k n-bit integers in-place using a Bitonic sorting network.

    For a given input list $x_1, x_2, \ldots, x_k$, applies the transform

        $$ |x_1, x_2, \ldots, x_k\rangle \mapsto |y_1, y_2, \ldots, y_k\rangle|\mathsf{junk}\rangle $$

    where $y_1, y_2, \ldots, y_k = \mathrm{sorted}(x_1, x_2, \ldots, x_k$, and the junk register
    stores the result of comparisons done during the sorting. Note that the junk register will
    be entangled with the input list register.

    Currently only supports $k$ being a power of two (#1090).

    The bitonic sorting network requires $\frac{k}{2} \frac{\log{k} (\log{k} + 1)}{2}$ total comparisons,
    and has depth $\frac{\log{k} (\log{k} + 1)}{2}$, when $k$ is a power of 2. Each comparison generates
    one ancilla qubit, so the total size of junk register equals the number of comparisons.

    Args:
        k: Number of integers to sort.
        bitsize: number of bits $n$ of each input number.

    Registers:
        xs: List of k integers we want to sort.
        junk (RIGHT): the generated ancilla qubits of each comparison in the sorting network.

    References:
        [Improved techniques for preparing eigenstates of fermionic Hamiltonians](https://www.nature.com/articles/s41534-018-0071-5).
        Supporting Information Sec. II.
    """
    k: SymbolicInt
    bitsize: SymbolicInt

    def __attrs_post_init__(self):
        k = self.k
        if not is_symbolic(k):
            assert not isinstance(k, sympy.Expr)
            assert k >= 1, f"length of input list must be positive, got {k=}"
            # TODO(#1090) support non-power-of-two input lengths
            assert (k & (k - 1)) == 0, f"length of input list must be a power of 2, got {k=}"

    @property
    def signature(self):
        return Signature(
            [
                Register("xs", QUInt(self.bitsize), shape=(self.k,)),
                Register("junk", QBit(), shape=(self.num_comparisons,), side=Side.RIGHT),
            ]
        )

    @cached_property
    def num_comparisons(self) -> SymbolicInt:
        """Number of `Comparator` gates used in the decomposition"""
        logk = bit_length(self.k - 1)
        return (self.k // 2) * ((logk * (logk + 1)) // 2)

    def pretty_name(self) -> str:
        return "BSort"

    def is_symbolic(self):
        return is_symbolic(self.bitsize, self.k)

    def build_composite_bloq(self, bb: 'BloqBuilder', xs: 'SoquetT') -> dict[str, 'SoquetT']:
        if self.is_symbolic():
            raise DecomposeTypeError(f"Cannot decompose symbolic {self=}")

        assert isinstance(xs, np.ndarray)

        if self.k == 1:
            return {'xs': xs, 'junk': np.array([])}

        if self.k == 2:
            xs[0], xs[1], anc = bb.add(Comparator(self.bitsize), a=xs[0], b=xs[1])
            return {'xs': xs, 'junk': np.array([anc])}

        xs_left, junk_left = bb.add(BitonicSort(self.k // 2, self.bitsize), xs=xs[: self.k // 2])
        xs_right, junk_right = bb.add(BitonicSort(self.k // 2, self.bitsize), xs=xs[self.k // 2 :])
        xs, junk_merge = bb.add(BitonicMerge(self.k // 2, self.bitsize), xs=xs_left, ys=xs_right)

        junk = np.concatenate([junk_left, junk_right, junk_merge])
        assert len(junk) == self.num_comparisons

        return {'xs': xs, 'junk': junk}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> set['BloqCountT']:
        return {(Comparator(self.bitsize), self.num_comparisons)}


@bloq_example
def _bitonic_sort() -> BitonicSort:
    bitonic_sort = BitonicSort(8, 4)
    return bitonic_sort


@bloq_example
def _bitonic_sort_symb() -> BitonicSort:
    n = sympy.Symbol('n')
    bitonic_sort_symb = BitonicSort(4, n)
    return bitonic_sort_symb


_BITONIC_SORT_DOC = BloqDocSpec(bloq_cls=BitonicSort, examples=[_bitonic_sort, _bitonic_sort_symb])
