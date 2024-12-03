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
from collections import Counter

import numpy as np
import sympy
from attrs import frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    QBit,
    QUInt,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.arithmetic import AddK, Equals, Xor
from qualtran.bloqs.arithmetic.lists import SymmetricDifference
from qualtran.bloqs.basic_gates import CNOT, ZeroEffect, ZeroState
from qualtran.bloqs.mcmt import And
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.symbolics import SymbolicInt

from .kxor_instance import KXorInstance


@frozen
class ColumnOfKthNonZeroEntry(Bloq):
    r"""Given $(S, k)$, compute the column of the $k$-th non-zero entry in row $S$.

    If the output is denoted as $f(S, k)$, then this bloq maps
    $(S, k, z, b)$ to $(S, k, z \oplus f'(S, k), b \oplus (k \ge s))$.
    where $s$ is the sparsity, and $f'(S, k)$ is by extending $f$
    such that for all $k \ge s$, $f'(S, k) = k$.
    Using $f'$ ensures the computation is reversible.
    Note: we must use the same extension $f'$ for both oracles.

    This algorithm is described by the following pseudo-code:
    ```
    def forward(S, k) -> f_S_k:
        nnz := 0 # counter
        for j in range(\bar{m}):
            T := S \Delta U_j
            if |T| == l:
                nnz := nnz + 1
                if nnz == k:
                    f_S_k ^= T
    ```

    Args:
        inst: the kXOR instance $\mathcal{I}$.
        ell: Kikuchi parameter $\ell$.

    Registers:
        S: index register to store $S \in {[n] \choose \ell}$.
        k: non-zero entry index register
        T: index register to store output $T = f(S, k) \in {[n] \choose \ell}$.
    """

    inst: KXorInstance
    ell: SymbolicInt

    @property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('S', self.index_dtype, shape=(self.ell,)),
                Register('k', self.index_dtype, shape=(self.ell,)),
                Register('T', self.index_dtype, shape=(self.ell,)),
                Register('flag', QBit()),
            ]
        )

    @property
    def index_dtype(self) -> QUInt:
        return QUInt(self.inst.index_bitsize)

    def adjoint(self) -> 'ColumnOfKthNonZeroEntry':
        return self

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        m = self.inst.num_unique_constraints
        ell, k = self.ell, self.inst.k

        counts_forward = Counter[Bloq]()

        # compute symmetric differences for each constraint
        counts_forward[SymmetricDifference(ell, k, ell, self.index_dtype)] += m

        # counter
        counts_forward[AddK(self.index_dtype, 1).controlled()] += m

        # compare counter each time
        counts_forward[Equals(self.index_dtype)] += m

        # when counter is equal (and updated in this iteration), we can copy the result
        counts_forward[And()] += m
        counts_forward[CNOT()] += m  # flip the final flag (flipped at most once)

        ### all counts
        counts = Counter[Bloq]()

        # copy the index (controlled by the final flag)
        counts[Xor(self.index_dtype).controlled()] += m

        # if nothing matched (final flag = 0), copy k and flip the flag bit
        counts[Xor(self.index_dtype).controlled()] += 1
        counts[Xor(QBit())] += 1

        for bloq, nb in counts_forward.items():
            # compute and uncompute all intermediate values.
            counts[bloq] += nb
            counts[bloq.adjoint()] += nb

        return counts


@bloq_example
def _col_kth_nz() -> ColumnOfKthNonZeroEntry:
    from qualtran.bloqs.optimization.k_xor_sat.kxor_instance import example_kxor_instance

    inst = example_kxor_instance()
    ell = 8

    col_kth_nz = ColumnOfKthNonZeroEntry(inst, ell)
    return col_kth_nz


@bloq_example
def _col_kth_nz_symb() -> ColumnOfKthNonZeroEntry:
    n, m, k, c, s = sympy.symbols("n m k c s", positive=True, integer=True)
    inst = KXorInstance.symbolic(n=n, m=m, k=k)
    ell = c * k

    col_kth_nz_symb = ColumnOfKthNonZeroEntry(inst, ell)
    return col_kth_nz_symb


@frozen
class IndexOfNonZeroColumn(Bloq):
    r"""Given $(S, T)$, compute $k$ such that $T$ is the $k$-th non-zero entry in row $S$.

    If $f(S, k)$ denotes the $k$-th non-zero entry in row $S$,
    then this bloq maps $(S, f'(S, k), z, b)$ to $(S, f'(S, k), z \oplus k, b \oplus )$.
    where $s$ is the sparsity, and $f'(S, k)$ is by extending $f$
    such that for all $k \ge s$, $f'(S, k) = k$.
    Using $f'$ ensures the computation is reversible.
    Note: we must use the same extension $f'$ for both oracles.

    This algorithm is described by the following pseudo-code:
    ```
    def reverse(S, f_S_k) -> k:
        nnz := 0 # counter
        for j in range(\bar{m}):
            T := S \Delta U_j
            if |T| == l:
                nnz := nnz + 1
            if T == f_S_k:
                k ^= nnz
    ```

    Args:
        inst: the kXOR instance $\mathcal{I}$.
        ell: Kikuchi parameter $\ell$.

    Registers:
        S: index register to store $S \in {[n] \choose \ell}$.
        k: non-zero entry index register
    """

    inst: KXorInstance
    ell: SymbolicInt

    @property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('S', self.index_dtype, shape=(self.ell,)),
                Register('k', self.index_dtype, shape=(self.ell,)),
                Register('T', self.index_dtype, shape=(self.ell,)),
                Register('flag', QBit()),
            ]
        )

    @property
    def index_dtype(self) -> QUInt:
        return QUInt(self.inst.index_bitsize)

    def adjoint(self) -> 'IndexOfNonZeroColumn':
        return self

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        m = self.inst.num_unique_constraints
        ell, k = self.ell, self.inst.k

        counts_forward = Counter[Bloq]()

        # compute symmetric differences for each constraint
        counts_forward[SymmetricDifference(ell, k, ell, self.index_dtype)] += m

        # counter
        counts_forward[AddK(self.index_dtype, 1).controlled()] += m

        # compare T to f_S_k each time
        counts_forward[Equals(self.index_dtype)] += m

        # when T is equal (and counter is updated in this iteration), we can copy the result
        counts_forward[And()] += m
        counts_forward[CNOT()] += m  # flip the final flag (flipped at most once)

        ### all counts
        counts = Counter[Bloq]()

        # copy the value of nnz (when final flag = 1)
        counts[Xor(self.index_dtype).controlled()] += m

        # if nothing matched (final flag = 0), copy k and flip the flag bit
        counts[Xor(self.index_dtype).controlled()] += 1
        counts[Xor(QBit())] += 1

        for bloq, nb in counts_forward.items():
            # compute and uncompute all intermediate values.
            counts[bloq] += nb
            counts[bloq.adjoint()] += nb

        return counts


@frozen
class KikuchiNonZeroIndex(Bloq):
    r"""Adjacency list oracle $O_F$ for the Kikuchi matrix.

    The oracle $O_F$ (Definition 4.5) takes in $i, k$,
    and outputs $i, f(i, k)$ where $f(i, k)$ is
    index of the $k$-th non-zero entry in row $i$.

    As the Kikuchi matrix is symmetric, we can use the same oracle for both rows and columns.

    The Kikuchi matrix is indexed by $S \in {[n] \choose k}$.
    For a given row $S$ and column $T$, the entry $\mathcal{K}_{k}_{S, T}$
    is potentially non-zero if $S \Delta T = U_j$ for some $j$, which is
    equivalent to $T = S \Delta U_j$.
    Here, $U_j$ is the $j$-th unique scope in the instance $\mathcal{I}$.

    To find the $k$-th non-zero entry, we use two oracles:
    1. $(S, k) \mapsto f(S, k)$, implemented by `ColumnOfKthNonZeroEntry`
    2. $(S, f(S, k)) \mapsto k$, implemented by `IndexOfNonZeroColumn`.

    Both these above oracles are unitary: they do not have any entangled ancilla/junk registers.


    Note on sparsity: This bloq expects the user to provide the sparsity, as it is in general
    difficult to compute the precise sparsity of the Kikuchi matrix efficiently. As long as the
    provided number is at least the true sparsity, the algorithm will work as expected.
    In case the provides sparsity is smaller, it is equivalent to making the remaining entries zero in the final block encoding.

    Args:
        inst: the kXOR instance $\mathcal{I}$.
        ell: Kikuchi parameter $\ell$.
        s: sparsity, i.e. max number of non-zero entries in a row/column.

    Registers:
        i: integer in [2^N]
        k: integer in [2^N]

    References:
        [Quartic quantum speedups for planted inference](https://arxiv.org/abs/2406.19378v1)
        Theorem 4.17, proof para 4 (top of page 39).
    """

    inst: KXorInstance
    ell: SymbolicInt
    s: SymbolicInt

    @property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('S', self.index_dtype, shape=(self.ell,)),
                Register('k', self.index_dtype, shape=(self.ell,)),
            ]
        )

    @property
    def index_dtype(self) -> QUInt:
        return QUInt(self.inst.index_bitsize)

    def build_composite_bloq(
        self, bb: 'BloqBuilder', S: 'Soquet', k: 'Soquet'
    ) -> dict[str, 'SoquetT']:
        T = np.array([bb.allocate(dtype=self.index_dtype) for _ in range(int(self.ell))])
        flag = bb.add(ZeroState())
        S, k, T, flag = bb.add(
            ColumnOfKthNonZeroEntry(self.inst, self.ell), S=S, k=k, T=T, flag=flag
        )
        S, T, k, flag = bb.add(IndexOfNonZeroColumn(self.inst, self.ell), S=S, T=T, k=k, flag=flag)
        for soq in k:
            bb.free(soq)
        bb.add(ZeroEffect(), q=flag)
        return dict(S=S, k=T)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {
            ColumnOfKthNonZeroEntry(self.inst, self.ell): 1,
            IndexOfNonZeroColumn(self.inst, self.ell): 1,
        }


@bloq_example
def _kikuchi_nonzero_index() -> KikuchiNonZeroIndex:
    from qualtran.bloqs.optimization.k_xor_sat.kxor_instance import example_kxor_instance

    inst = example_kxor_instance()
    ell = 8
    s = inst.brute_force_sparsity(ell)

    kikuchi_nonzero_index = KikuchiNonZeroIndex(inst, ell, s=s)
    return kikuchi_nonzero_index


@bloq_example
def _kikuchi_nonzero_index_symb() -> KikuchiNonZeroIndex:
    from qualtran.bloqs.optimization.k_xor_sat.kxor_instance import KXorInstance

    n, m, k, c, s = sympy.symbols("n m k c s", positive=True, integer=True)
    inst = KXorInstance.symbolic(n=n, m=m, k=k)
    ell = c * k

    kikuchi_nonzero_index_symb = KikuchiNonZeroIndex(inst, ell, s=s)
    return kikuchi_nonzero_index_symb


_KIKUCHI_NONZERO_INDEX_DOC = BloqDocSpec(
    bloq_cls=KikuchiNonZeroIndex, examples=[_kikuchi_nonzero_index]
)
