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
from typing import cast, Iterable, Sequence, Set, TYPE_CHECKING, TypeAlias, Union

from attrs import field, frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqDocSpec,
    BQUInt,
    DecomposeTypeError,
    QBit,
    QUInt,
    Signature,
    Soquet,
)
from qualtran.bloqs.arithmetic.bitwise import XorK
from qualtran.bloqs.arithmetic.comparison import EqualsAConstant
from qualtran.linalg.permutation import (
    CycleT,
    decompose_permutation_into_cycles,
    decompose_permutation_map_into_cycles,
)
from qualtran.symbolics import bit_length, is_symbolic, Shaped, slen, SymbolicInt

if TYPE_CHECKING:
    import sympy

    from qualtran import BloqBuilder, SoquetT
    from qualtran.resource_counting import BloqCountDictT, BloqCountT, SympySymbolAllocator

SymbolicCycleT: TypeAlias = Union[CycleT, Shaped]


def _convert_cycle(cycle) -> Union[tuple[int, ...], Shaped]:
    if isinstance(cycle, Shaped):
        return cycle
    return tuple(cycle)


@frozen
class PermutationCycle(Bloq):
    r"""Apply a single permutation cycle on the basis states.

    Given a permutation cycle $C = (v_0 v_2 \ldots v_{k - 1})$, applies the following unitary:

        $$
            U|v_i\rangle \mapsto |v_{(i + 1)\mod k}\rangle
        $$

    for each $i \in [0, k)$, and

        $$
            U|x\rangle \mapsto |x\rangle
        $$

    and for every $x \not\in C$.

    Args:
        N: the total size the permutation acts on.
        cycle: the permutation cycle to apply.

    Registers:
        x: integer register storing a value in [0, ..., N - 1]

    References:
        [A simple quantum algorithm to efficiently prepare sparse states](https://arxiv.org/abs/2310.19309v1)
        Appendix B, Algorithm 7.
    """

    N: SymbolicInt
    cycle: Union[tuple[int, ...], Shaped] = field(converter=_convert_cycle)

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(x=BQUInt(self.bitsize, self.N))

    @cached_property
    def bitsize(self):
        return bit_length(self.N - 1)

    def build_composite_bloq(self, bb: 'BloqBuilder', x: 'SoquetT') -> dict[str, 'SoquetT']:
        if is_symbolic(self.cycle):
            raise DecomposeTypeError(f"cannot decompose symbolic {self}")

        a: 'SoquetT' = bb.allocate(dtype=QBit())

        for k, x_k in enumerate(self.cycle):
            x, a = bb.add_t(EqualsAConstant(self.bitsize, x_k), x=x, target=a)

            delta = x_k ^ self.cycle[(k + 1) % len(self.cycle)]
            a, x = bb.add_t(XorK(QUInt(self.bitsize), delta).controlled(), ctrl=a, x=x)

        x, a = bb.add_t(EqualsAConstant(self.bitsize, self.cycle[0]), x=x, target=a)

        bb.free(cast(Soquet, a))

        return {'x': x}

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        if is_symbolic(self.cycle):
            x = ssa.new_symbol('x')
            cycle_len = slen(self.cycle)
            return {
                EqualsAConstant(self.bitsize, x): cycle_len + 1,
                XorK(QUInt(self.bitsize), x).controlled(): cycle_len,
            }

        return super().build_call_graph(ssa)


@bloq_example
def _permutation_cycle() -> PermutationCycle:
    permutation_cycle = PermutationCycle(4, (0, 1, 2))
    return permutation_cycle


@bloq_example
def _permutation_cycle_symb_N() -> PermutationCycle:
    import sympy

    N = sympy.symbols("n", positive=True, integer=True)
    cycle = (3, 1, 2)
    permutation_cycle_symb_N = PermutationCycle(N, cycle)
    return permutation_cycle_symb_N


@bloq_example
def _permutation_cycle_symb() -> PermutationCycle:
    import sympy

    from qualtran.symbolics import Shaped

    N, L = sympy.symbols("N L", positive=True, integer=True)
    cycle = Shaped((L,))
    permutation_cycle_symb = PermutationCycle(N, cycle)
    return permutation_cycle_symb


_PERMUTATION_CYCLE_DOC = BloqDocSpec(
    bloq_cls=PermutationCycle,
    import_line='from qualtran.bloqs.arithmetic.permutation import PermutationCycle',
    examples=[_permutation_cycle_symb_N, _permutation_cycle_symb, _permutation_cycle],
)


def _convert_cycles(cycles) -> Union[tuple[SymbolicCycleT, ...], Shaped]:
    if isinstance(cycles, Shaped):
        return cycles
    return tuple(_convert_cycle(cycle) for cycle in cycles)


@frozen
class Permutation(Bloq):
    r"""Apply a permutation of [0, N - 1] on the basis states.

    Given a permutation $P : [0, N - 1] \to [0, N - 1]$, this bloq applies the unitary:

    $$
        U|x\rangle = |P(x)\rangle
    $$

    Decomposes a permutation into cycles and applies them in order.
    See :meth:`from_dense_permutation` to construct this bloq from a permutation,
    and :meth:`from_partial_permutation_map` to construct it from a mapping.

    Args:
        N: the total size the permutation acts on.
        cycles: a sequence of permutation cycles that form the permutation.

    Registers:
        x: integer register storing a value in [0, ..., N - 1]

    References:
        [A simple quantum algorithm to efficiently prepare sparse states](https://arxiv.org/abs/2310.19309v1)
        Appendix B.
    """

    N: SymbolicInt
    cycles: Union[tuple[SymbolicCycleT, ...], Shaped] = field(converter=_convert_cycles)

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(x=BQUInt(self.bitsize, self.N))

    @cached_property
    def bitsize(self):
        return bit_length(self.N - 1)

    def is_symbolic(self):
        return is_symbolic(self.N, self.cycles) or (
            isinstance(self.cycles, tuple) and is_symbolic(*self.cycles)
        )

    @classmethod
    def from_dense_permutation(cls, permutation: Sequence[int]):
        """Construct a permutation bloq from a dense permutation of size `N`.

        Args:
            permutation: a sequence of length `N` containing a permutation of `[0, N)`.
        """
        N = len(permutation)
        cycles = tuple(decompose_permutation_into_cycles(permutation))
        return cls(N, cycles)

    @classmethod
    def from_partial_permutation_map(cls, N: SymbolicInt, permutation_map: dict[int, int]):
        """Construct a permutation bloq from a (partial) permutation mapping

        Constructs a permuation of `[0, N)` from a partial mapping. Any numbers that
        do not occur in `permutation_map` (i.e. as keys or values) are treated as
        mapping to themselves.

        Args:
            N: the upper limit, i.e. permutation is on range `[0, N)`
            permutation_map: a dictionary defining the permutation
        """
        cycles = tuple(decompose_permutation_map_into_cycles(permutation_map))
        return cls(N, cycles)

    @classmethod
    def from_cycle_lengths(cls, N: SymbolicInt, cycle_lengths: Iterable[SymbolicInt]):
        """Construct a permutation bloq from a dense permutation of size `N`.

        Args:
            N: the upper limit, i.e. permutation is on range `[0, N)`
            cycle_lengths: a tuple of lengths of each non-trivial cycle (i.e. length at least 2).
        """
        cycles = tuple(
            Shaped((cycle_len,))
            for cycle_len in cycle_lengths
            if is_symbolic(cycle_len) or cycle_len >= 2
        )
        return cls(N, cycles)

    def build_composite_bloq(self, bb: 'BloqBuilder', x: 'Soquet') -> dict[str, 'SoquetT']:
        if is_symbolic(self.cycles):
            raise DecomposeTypeError(f"cannot decompose symbolic {self}")

        for cycle in self.cycles:
            x = bb.add(PermutationCycle(self.N, cycle), x=x)

        return {'x': x}

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        if is_symbolic(self.cycles):
            # worst case cost: single cycle of length N
            cycle = Shaped((self.N,))
            return {PermutationCycle(self.N, cycle): 1}

        return super().build_call_graph(ssa)


@bloq_example
def _permutation() -> Permutation:
    permutation = Permutation.from_dense_permutation([1, 3, 0, 2])
    return permutation


@bloq_example
def _permutation_symb() -> Permutation:
    import sympy

    from qualtran.symbolics import Shaped

    N, k = sympy.symbols("N k", positive=True, integer=True)
    permutation_symb = Permutation(N, Shaped((k,)))
    return permutation_symb


@bloq_example
def _permutation_symb_with_cycles() -> Permutation:
    import sympy

    from qualtran.symbolics import Shaped

    N = sympy.symbols("N", positive=True, integer=True)
    n_cycles = 4
    d = sympy.IndexedBase('d', shape=(n_cycles,))
    permutation_symb_with_cycles = Permutation(N, tuple(Shaped((d[i],)) for i in range(n_cycles)))
    return permutation_symb_with_cycles


@bloq_example
def _sparse_permutation() -> Permutation:
    sparse_permutation = Permutation.from_partial_permutation_map(
        16, {0: 1, 1: 3, 2: 8, 3: 15, 4: 12}
    )
    return sparse_permutation


@bloq_example
def _sparse_permutation_with_symbolic_N() -> Permutation:
    import sympy

    N = sympy.symbols("N", positive=True, integer=True)
    sparse_permutation_with_symbolic_N = Permutation.from_partial_permutation_map(
        N, {0: 1, 1: 3, 2: 4, 3: 7}
    )
    return sparse_permutation_with_symbolic_N


_PERMUTATION_DOC = BloqDocSpec(
    bloq_cls=Permutation,
    import_line='from qualtran.bloqs.arithmetic.permutation import Permutation',
    examples=[
        _permutation,
        _permutation_symb,
        _permutation_symb_with_cycles,
        _sparse_permutation,
        _sparse_permutation_with_symbolic_N,
    ],
)
