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
from typing import cast, Sequence, Set, TYPE_CHECKING, TypeAlias, Union

from attrs import field, frozen

from qualtran import (
    Bloq,
    bloq_example,
    BoundedQUInt,
    DecomposeTypeError,
    QBit,
    QUInt,
    Signature,
    Soquet,
)
from qualtran.bloqs.arithmetic.addition import XorK
from qualtran.bloqs.arithmetic.comparison import EqualsAConstant
from qualtran.linalg.permutation import (
    CycleT,
    decompose_permutation_into_cycles,
    decompose_sparse_prefix_permutation_into_cycles,
)
from qualtran.symbolics import bit_length, is_symbolic, Shaped, slen, SymbolicInt

if TYPE_CHECKING:
    import sympy

    from qualtran import BloqBuilder, SoquetT
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


SymbolicCycleT: TypeAlias = Union[CycleT, Shaped]


def _convert_cycle(cycle) -> Union[tuple[int, ...], Shaped]:
    if isinstance(cycle, Shaped):
        return cycle
    return tuple(cycle)


@frozen
class PermutationCycle(Bloq):
    r"""Apply a single permutation cycle on the basis states.

    Args:
        N: the total size the permutation acts on.
        cycle: the permutation cycle to apply.

    Registers:
        q: integer register storing a value in [0, ..., N - 1]

    References:
        [A simple quantum algorithm to efficiently prepare sparse states](https://arxiv.org/abs/2310.19309v1)
        Appendix B, Algorithm 7.
    """

    N: SymbolicInt
    cycle: Union[tuple[int, ...], Shaped] = field(converter=_convert_cycle)

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(q=BoundedQUInt(self.bitsize, self.N))

    @cached_property
    def bitsize(self):
        return bit_length(self.N - 1)

    def is_symbolic(self):
        return is_symbolic(self.N, self.cycle)

    def build_composite_bloq(self, bb: 'BloqBuilder', q: 'SoquetT') -> dict[str, 'SoquetT']:
        if self.is_symbolic():
            raise DecomposeTypeError(f"cannot decompose symbolic {self}")
        assert not isinstance(self.cycle, Shaped)

        a: 'SoquetT' = bb.allocate(dtype=QBit())

        for k, x_k in enumerate(self.cycle):
            q, a = bb.add_t(EqualsAConstant(self.bitsize, x_k), x=q, target=a)

            delta = x_k ^ self.cycle[(k + 1) % len(self.cycle)]
            a, q = bb.add_t(XorK(QUInt(self.bitsize), delta).controlled(), ctrl=a, x=q)

        q, a = bb.add_t(EqualsAConstant(self.bitsize, self.cycle[0]), x=q, target=a)

        bb.free(cast(Soquet, a))

        return {'q': q}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        if self.is_symbolic():
            x = ssa.new_symbol('x')
            cycle_len = slen(self.cycle)
            return {
                (EqualsAConstant(self.bitsize, x), cycle_len + 1),
                (XorK(QUInt(self.bitsize), x).controlled(), cycle_len),
            }

        return super().build_call_graph(ssa)


@bloq_example
def _permutation_cycle() -> PermutationCycle:
    permutation_cycle = PermutationCycle(4, (0, 1, 2))
    return permutation_cycle


@bloq_example
def _permutation_cycle_symb() -> PermutationCycle:
    import sympy

    N, L = sympy.symbols("N L")
    cycle = Shaped((L,))
    permutation_cycle_symb = PermutationCycle(N, cycle)
    return permutation_cycle_symb


def _convert_cycles(cycles) -> Union[tuple[SymbolicCycleT, ...], Shaped]:
    if isinstance(cycles, Shaped):
        return cycles
    return tuple(_convert_cycle(cycle) for cycle in cycles)


@frozen
class Permutation(Bloq):
    """Apply a permutation of [0, N - 1] on the basis states.

    Decomposes a permutation into cycles and applies them in order.
    See :meth:`from_dense_permutation` to construct this bloq from a permutation.

    Args:
        N: the total size the permutation acts on.
        cycles: a sequence of permutation cycles that form the permutation.

    Registers:
        q: integer register storing a value in [0, ..., N - 1]

    References:
        [A simple quantum algorithm to efficiently prepare sparse states](https://arxiv.org/abs/2310.19309v1)
        Appendix B.
    """

    N: SymbolicInt
    cycles: Union[tuple[SymbolicCycleT, ...], Shaped] = field(converter=_convert_cycles)

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(q=BoundedQUInt(self.bitsize, self.N))

    @cached_property
    def bitsize(self):
        return bit_length(self.N - 1)

    def is_symbolic(self):
        return is_symbolic(self.N, self.cycles) or (
            isinstance(self.cycles, tuple) and is_symbolic(*self.cycles)
        )

    @classmethod
    def from_dense_permutation(cls, permutation: Sequence[int]):
        N = len(permutation)
        cycles = tuple(decompose_permutation_into_cycles(permutation))
        return cls(N, cycles)

    @classmethod
    def from_sparse_permutation_prefix(cls, N: int, permutation_prefix: Sequence[int]):
        cycles = tuple(decompose_sparse_prefix_permutation_into_cycles(permutation_prefix, N))
        return cls(N, cycles)

    @classmethod
    def from_cycle_lengths(cls, N: SymbolicInt, cycle_lengths: tuple[SymbolicInt, ...]):
        cycles = tuple(Shaped((cycle_len,)) for cycle_len in cycle_lengths)
        return cls(N, cycles)

    def build_composite_bloq(self, bb: 'BloqBuilder', q: 'Soquet') -> dict[str, 'SoquetT']:
        if self.is_symbolic():
            raise DecomposeTypeError(f"cannot decompose symbolic {self}")

        assert not isinstance(self.cycles, Shaped)
        for cycle in self.cycles:
            q = bb.add(PermutationCycle(self.N, cycle), q=q)

        return {'q': q}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        if self.is_symbolic():
            # worst case cost: single cycle of length N
            cycle = Shaped((self.N,))
            return {(PermutationCycle(self.N, cycle), 1)}

        return super().build_call_graph(ssa)


@bloq_example
def _permutation() -> Permutation:
    permutation = Permutation.from_dense_permutation([1, 3, 0, 2])
    return permutation


@bloq_example
def _permutation_symb() -> Permutation:
    import sympy

    N, k = sympy.symbols("N k")
    permutation_symb = Permutation(N, Shaped((k,)))
    return permutation_symb


@bloq_example
def _permutation_symb_with_cycles() -> Permutation:
    import sympy

    N = sympy.symbols("N")
    n_cycles = 4
    d = sympy.IndexedBase('d', shape=(n_cycles,))
    permutation_symb = Permutation(N, tuple(Shaped((d[i],)) for i in range(n_cycles)))
    return permutation_symb


@bloq_example
def _sparse_permutation() -> Permutation:
    sparse_permutation = Permutation.from_sparse_permutation_prefix(16, [1, 3, 8, 15, 12])
    return sparse_permutation
