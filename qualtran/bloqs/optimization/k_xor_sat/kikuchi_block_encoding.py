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
"""Section 4.4.2 Simulating the Kikuchi Hamiltonian

This module contains oracles to implement the block-encoding of the Kikuchi
Hamiltonian corresponding to an input k-XOR-SAT instance.

References:
    [Quartic quantum speedups for planted inference](https://arxiv.org/abs/2406.19378v1)
    Section 4.4.2 for algorithm. Section 2.4 for definitions and notation.
"""
from functools import cached_property

import sympy
from attrs import field, frozen

from qualtran import (
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    BQUInt,
    QAny,
    QBit,
    QUInt,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.block_encoding import BlockEncoding
from qualtran.bloqs.block_encoding.sparse_matrix import RowColumnOracle
from qualtran.bloqs.block_encoding.sparse_matrix_hermitian import (
    SparseMatrixHermitian,
    SqrtEntryOracle,
)
from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.symbolics import is_symbolic, SymbolicFloat, SymbolicInt

from .kikuchi_adjacency_list import KikuchiNonZeroIndex
from .kikuchi_adjacency_matrix import KikuchiMatrixEntry
from .kxor_instance import KXorInstance


@frozen
class BlackBoxKikuchiEntryOracle(SqrtEntryOracle):
    r"""Wrapper around the adjacency matrix oracle $O_H$ of the Kikuchi graph."""

    O_H: KikuchiMatrixEntry

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(
            q=QBit(), i=QAny(self.system_bitsize), j=QAny(self.system_bitsize)
        )

    @property
    def system_bitsize(self) -> SymbolicInt:
        return self.O_H.composite_index_bitsize

    @property
    def epsilon(self) -> SymbolicFloat:
        """precision due to fixed-point approximation of entries.

        In the good case, whp (i.e. 1 - o(1)), the entries are in [-2, 2],
        whose corresponding angles can be represented exactly with 3 bits.
        I.e. `arccos(sqrt(x / 2)) / pi` for `x in [-2, 2]` are `2, 1.5, 1, 0.5, 0`.
        """
        return 0

    @property
    def _phasegrad_bitsize(self) -> SymbolicInt:
        return self.O_H.entry_bitsize

    def build_composite_bloq(
        self, bb: 'BloqBuilder', q: 'Soquet', i: 'Soquet', j: 'Soquet'
    ) -> dict[str, 'SoquetT']:
        i, j, q = bb.add(self.O_H, S=i, T=j, q=q)
        return dict(q=q, i=i, j=j)


@frozen
class BlackBoxKikuchiRowColumnOracle(RowColumnOracle):
    r"""Wrapper around the adjacency list oracle $O_F$ of the Kikuchi graph."""

    O_F: KikuchiNonZeroIndex

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(
            l=BQUInt(self.system_bitsize, self.num_nonzero), i=QUInt(self.system_bitsize)
        )

    @property
    def system_bitsize(self) -> SymbolicInt:
        return self.O_F.index_dtype.num_qubits * self.O_F.ell

    @property
    def num_nonzero(self) -> SymbolicInt:
        return self.O_F.s

    def build_composite_bloq(
        self, bb: 'BloqBuilder', l: 'Soquet', i: 'Soquet'
    ) -> dict[str, 'SoquetT']:
        i, l = bb.add(self.O_F, S=i, k=l)
        return dict(l=l, i=i)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {self.O_F: 1}


@frozen
class KikuchiHamiltonian(BlockEncoding):
    r"""Block encoding of the Kikuchi matrix $\mathcal{K}_\ell$.

    This is implemented by a sparse matrix block encoding using the adjacency matrix
    and adjacency list oracles.

    This assumes a default sparsity of $\bar{m}$, which is the number of unique
    scopes in the instance $\mathcal{I}$.
    If a better bound on sparsity is known, it can be passed in by the user.

    Args:
        inst: kXOR instance $\mathcal{I}$.
        ell: Kikuchi parameter $\ell$.
        entry_bitsize: Number of bits $b$ to approximate the matrix entries (angles) to.
        s: sparsity of the Kikuchi matrix, defaults to $\bar{m}$.
    """

    inst: KXorInstance
    ell: SymbolicInt
    entry_bitsize: SymbolicInt = field()
    s: SymbolicInt = field()

    @s.default
    def _default_sparsity(self) -> SymbolicInt:
        return self.inst.num_unique_constraints

    @entry_bitsize.default
    def _default_entry_bitsize(self):
        if is_symbolic(self.inst.max_rhs) or self.inst.max_rhs == 2:
            # one T gate suffices!
            return 3
        raise ValueError("Entries outside range [-2, 2], please specify an explicit entry_bitsize.")

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(
            system=self.system_bitsize, ancilla=self.ancilla_bitsize, resource=self.resource_bitsize
        )

    @cached_property
    def _sparse_matrix_encoding(self) -> SparseMatrixHermitian:
        blackbox_O_F = BlackBoxKikuchiRowColumnOracle(self.oracle_O_F)
        blackbox_O_H = BlackBoxKikuchiEntryOracle(self.oracle_O_H)
        return SparseMatrixHermitian(
            col_oracle=blackbox_O_F, entry_oracle=blackbox_O_H, eps=blackbox_O_H.epsilon
        )

    @cached_property
    def oracle_O_H(self) -> KikuchiMatrixEntry:
        r"""Maps $|i, j\rangle |0\rangle$ to $|i, j\rangle (\sqrt{A_{ij}} |0\rangle + \sqrt{1 - |A_{ij}|} |1\rangle)"""
        return KikuchiMatrixEntry(inst=self.inst, ell=self.ell, entry_bitsize=self.entry_bitsize)

    @cached_property
    def oracle_O_F(self) -> KikuchiNonZeroIndex:
        r"""Maps `i, k` to `i, f(i, k)` where `f(i, k)` is the column of the `k`-th nonzero entry in row `i`."""
        return KikuchiNonZeroIndex(inst=self.inst, ell=self.ell, s=self.s)

    @property
    def alpha(self) -> SymbolicFloat:
        return self._sparse_matrix_encoding.alpha

    @property
    def system_bitsize(self) -> SymbolicInt:
        return self._sparse_matrix_encoding.system_bitsize

    @property
    def ancilla_bitsize(self) -> SymbolicInt:
        return self._sparse_matrix_encoding.ancilla_bitsize

    @property
    def resource_bitsize(self) -> SymbolicInt:
        return self._sparse_matrix_encoding.resource_bitsize

    @property
    def epsilon(self) -> SymbolicFloat:
        return self._sparse_matrix_encoding.epsilon

    @property
    def signal_state(self) -> BlackBoxPrepare:
        return self._sparse_matrix_encoding.signal_state

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> dict[str, 'SoquetT']:
        return bb.add_d(self._sparse_matrix_encoding, **soqs)

    def __str__(self):
        return 'B[K_l]'


@bloq_example
def _kikuchi_matrix() -> KikuchiHamiltonian:
    from qualtran.bloqs.optimization.k_xor_sat.kxor_instance import example_kxor_instance

    inst = example_kxor_instance()
    ell = 8

    kikuchi_matrix = KikuchiHamiltonian(inst, ell)
    return kikuchi_matrix


@bloq_example
def _kikuchi_matrix_symb() -> KikuchiHamiltonian:
    from qualtran.bloqs.optimization.k_xor_sat.kxor_instance import KXorInstance

    n, m, k, c = sympy.symbols("n m k c", positive=True, integer=True)
    inst = KXorInstance.symbolic(n=n, m=m, k=k)
    ell = c * k

    kikuchi_matrix_symb = KikuchiHamiltonian(inst, ell)
    return kikuchi_matrix_symb


_KIKUCHI_HAMILTONIAN_DOC = BloqDocSpec(
    bloq_cls=KikuchiHamiltonian, examples=[_kikuchi_matrix, _kikuchi_matrix_symb]
)
