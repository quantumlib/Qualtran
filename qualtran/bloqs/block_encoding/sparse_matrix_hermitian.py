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
import abc
from collections import Counter
from functools import cached_property

import attrs
import numpy as np
import sympy
from attrs import frozen

from qualtran import (
    AddControlledT,
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    CtrlSpec,
    DecomposeTypeError,
    QAny,
    QBit,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.basic_gates import CSwap, Ry, Swap
from qualtran.bloqs.block_encoding import BlockEncoding
from qualtran.bloqs.block_encoding.sparse_matrix import RowColumnOracle
from qualtran.bloqs.bookkeeping import Partition
from qualtran.bloqs.bookkeeping.auto_partition import AutoPartition, Unused
from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity
from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare
from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
    PrepareUniformSuperposition,
)
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.resource_counting.generalizers import ignore_split_join
from qualtran.symbolics import is_symbolic, SymbolicFloat, SymbolicInt
from qualtran.symbolics.math_funcs import bit_length


class SqrtEntryOracle(Bloq):
    r"""Oracle specifying the sqrt of entries of a sparse-access matrix.

    In the reference, this is the interface of
    $$O_A : \ket{0}\ket{i}\ket{j} \mapsto (\sqrt{A_{ij}} \ket{0} + \sqrt{1 - |A_{ij}|}\ket{i}\ket{j}).$$

    Registers:
        q: The flag qubit that is rotated proportionally to the value of the entry.
        i: The row index.
        j: The column index.

    References:
        [Lecture Notes on Quantum Algorithms for Scientific Computation](https://arxiv.org/abs/2201.08309). Lin Lin (2022). Ch. 6.5.
    """

    @property
    @abc.abstractmethod
    def system_bitsize(self) -> SymbolicInt:
        """The number of bits used to represent an index."""

    @property
    @abc.abstractmethod
    def epsilon(self) -> SymbolicFloat:
        """The number of bits used to represent an index."""

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(
            q=QBit(), i=QAny(self.system_bitsize), j=QAny(self.system_bitsize)
        )


@frozen
class SparseMatrixHermitian(BlockEncoding):
    r"""Hermitian Block encoding of a sparse-access Hermitian matrix.

    Given column and entry oracles $O_c$ and $O_A$ for an $s$-sparse Hermitian matrix
    $A \in \mathbb{C}^{2^n \times 2^n}$, i.e. one where each row / column has exactly $s$ non-zero
    entries, computes a $(s, n+1, \epsilon)$-block encoding of $A$ as follows:
    ```
               ┌────┐
    a     |0> ─┤    ├─     |0> ───────────────────────X────────────────────
               │    │           ┌──┐                  |               ┌──┐
               │ U  │  =        │ n│ ┌────┐ ┌────┐    | ┌────┐ ┌────┐ │ n│
    l   |0^n> ─┤  A ├─   |0^n> ─┤H ├─┤ O  ├─┤    ├─X──|─┤    ├─┤ O* ├─┤H ├─
               │    │           └──┘ |  c | │    │ |  | │    │ |  c | └──┘
               │    │                └────┘ │ O  │ │  | │ O* │ └────┘
    b     |0> ─┤    ├─     |0> ────────|────┤  A ├─|──X─┤  A ├───|─────────
               |    |                ┌────┐ |    | |    |    | ┌────┐
               |    |                | O  | |    | |    |    | | O* |
    j   |Psi> ─┤    ├─   |Psi> ──────┤  c ├─┤    ├─X────┤    ├─┤  c ├──────
               └────┘                └────┘ └────┘      └────┘ └────┘
    ```

    To encode a matrix of irregular dimension, the matrix should first be embedded into one of
    dimension $2^n \times 2^n$ for suitable $n$.
    To encode a matrix where each row / column has at most $s$ non-zero entries, some zeroes should
    be treated as if they were non-zero so that each row / column has exactly $s$ non-zero entries.

    For encoding a non-hermitian matrix, or a slightly more efficient (but non Hermitian-encoding)
    of a matrix, use :class:`SparseMatrix` instead.

    Args:
        col_oracle: The column oracle $O_c$. See `RowColumnOracle` for definition.
        entry_oracle: The entry oracle $O_A$. See `EntryOracle` for definition.
        eps: The precision of the block encoding.
        is_controlled: if True, returns the controlled block-encoding.

    Registers:
        ctrl: The single qubit control register. (present only if `is_controlled` is `True`)
        system: The system register.
        ancilla: The ancilla register.
        resource: The resource register (present only if `bitsize > 0`).

    References:
        [Lecture Notes on Quantum Algorithms for Scientific Computation](https://arxiv.org/abs/2201.08309).
        Lin Lin (2022). Ch. 6.5. Proposition 6.8, Fig 6.7.
    """

    col_oracle: RowColumnOracle
    entry_oracle: SqrtEntryOracle
    eps: SymbolicFloat
    is_controlled: bool = False

    def __attrs_post_init__(self):
        if self.col_oracle.system_bitsize != self.entry_oracle.system_bitsize:
            raise ValueError("column and entry oracles must have same bitsize")

    @cached_property
    def signature(self) -> Signature:
        n_ctrls = 1 if self.is_controlled else 0

        return Signature.build_from_dtypes(
            ctrl=QAny(n_ctrls),
            system=QAny(self.system_bitsize),
            ancilla=QAny(self.ancilla_bitsize),
            resource=QAny(self.resource_bitsize),  # if resource_bitsize is 0, not present
        )

    @cached_property
    def system_bitsize(self) -> SymbolicInt:
        return self.entry_oracle.system_bitsize

    def __str__(self) -> str:
        return "B[SparseMatrixHermitian]"

    @cached_property
    def alpha(self) -> SymbolicFloat:
        return self.col_oracle.num_nonzero

    @cached_property
    def ancilla_bitsize(self) -> SymbolicInt:
        return self.system_bitsize + 2

    @cached_property
    def resource_bitsize(self) -> SymbolicInt:
        return 0

    @cached_property
    def epsilon(self) -> SymbolicFloat:
        return self.eps

    @property
    def signal_state(self) -> BlackBoxPrepare:
        return BlackBoxPrepare(PrepareIdentity.from_bitsizes([self.ancilla_bitsize]))

    @cached_property
    def diffusion(self):
        unused = self.system_bitsize - bit_length(self.col_oracle.num_nonzero - 1)
        return AutoPartition(
            PrepareUniformSuperposition(n=self.col_oracle.num_nonzero),
            partitions=[
                (Register("target", QAny(self.system_bitsize)), [Unused(unused), "target"])
            ],
        )

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:
        counts = Counter[Bloq]()

        counts[self.diffusion] += 1
        counts[self.col_oracle] += 1
        counts[self.entry_oracle] += 1
        if self.is_controlled:
            counts[CSwap(self.system_bitsize)] += 1
            counts[CSwap(1)] += 1
        else:
            counts[Swap(self.system_bitsize)] += 1
            counts[Swap(1)] += 1
        counts[self.entry_oracle.adjoint()] += 1
        counts[self.col_oracle.adjoint()] += 1
        counts[self.diffusion.adjoint()] += 1

        return counts

    def build_composite_bloq(
        self, bb: BloqBuilder, system: SoquetT, ancilla: SoquetT, **soqs
    ) -> dict[str, SoquetT]:
        if is_symbolic(self.system_bitsize) or is_symbolic(self.col_oracle.num_nonzero):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self=}")

        ctrl = soqs.pop('ctrl', None)

        assert not isinstance(ancilla, np.ndarray)
        partition_ancilla = Partition(
            n=self.ancilla_bitsize,
            regs=(
                Register('a', QBit()),
                Register('l', QAny(self.system_bitsize)),
                Register('b', QBit()),
            ),
        )

        a, l, b = bb.add(partition_ancilla, x=ancilla)

        l = bb.add(self.diffusion, target=l)
        l, system = bb.add(self.col_oracle, l=l, i=system)
        b, l, system = bb.add(self.entry_oracle, q=b, i=l, j=system)

        if self.is_controlled:
            ctrl, l, system = bb.add(CSwap(self.system_bitsize), ctrl=ctrl, x=l, y=system)
            ctrl, a, b = bb.add(CSwap(1), ctrl=ctrl, x=a, y=b)
        else:
            l, system = bb.add(Swap(self.system_bitsize), x=l, y=system)
            a, b = bb.add(Swap(1), x=a, y=b)

        b, l, system = bb.add(self.entry_oracle.adjoint(), q=b, i=l, j=system)
        l, system = bb.add(self.col_oracle.adjoint(), l=l, i=system)
        l = bb.add(self.diffusion.adjoint(), target=l)

        ancilla = bb.add(partition_ancilla.adjoint(), a=a, l=l, b=b)

        out_soqs = {"system": system, "ancilla": ancilla}
        if self.is_controlled:
            out_soqs |= {"ctrl": ctrl}
        return out_soqs

    def adjoint(self) -> 'SparseMatrixHermitian':
        return self

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> tuple['Bloq', 'AddControlledT']:
        from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv_from_bloqs

        return get_ctrl_system_1bit_cv_from_bloqs(
            self,
            ctrl_spec,
            current_ctrl_bit=1 if self.is_controlled else None,
            bloq_with_ctrl=self if self.is_controlled else attrs.evolve(self, is_controlled=True),
            ctrl_reg_name='ctrl',
        )


@frozen
class UniformSqrtEntryOracle(SqrtEntryOracle):
    """Oracle specifying the entries of a matrix with uniform entries."""

    system_bitsize: SymbolicInt
    entry: float
    eps: float = 1e-11

    @property
    def epsilon(self) -> SymbolicFloat:
        return self.eps

    def build_composite_bloq(
        self, bb: BloqBuilder, q: Soquet, **soqs: SoquetT
    ) -> dict[str, SoquetT]:
        # Either Rx or Ry work here; Rx would induce a phase on the subspace with non-zero ancilla
        # See https://arxiv.org/abs/2302.10949 for reference that uses Rx
        soqs["q"] = bb.add(Ry(2 * np.arccos(np.sqrt(self.entry))), q=q)
        return soqs


@bloq_example(generalizer=ignore_split_join)
def _sparse_matrix_hermitian_block_encoding() -> SparseMatrixHermitian:
    from qualtran.bloqs.block_encoding.sparse_matrix import TopLeftRowColumnOracle
    from qualtran.bloqs.block_encoding.sparse_matrix_hermitian import UniformSqrtEntryOracle

    col_oracle = TopLeftRowColumnOracle(system_bitsize=2)
    entry_oracle = UniformSqrtEntryOracle(system_bitsize=2, entry=0.3)
    sparse_matrix_hermitian_block_encoding = SparseMatrixHermitian(col_oracle, entry_oracle, eps=0)
    return sparse_matrix_hermitian_block_encoding


@bloq_example
def _sparse_matrix_symb_hermitian_block_encoding() -> SparseMatrixHermitian:
    from qualtran.bloqs.block_encoding.sparse_matrix import TopLeftRowColumnOracle
    from qualtran.bloqs.block_encoding.sparse_matrix_hermitian import UniformSqrtEntryOracle

    n = sympy.Symbol('n', positive=True, integer=True)
    col_oracle = TopLeftRowColumnOracle(system_bitsize=n)
    entry_oracle = UniformSqrtEntryOracle(system_bitsize=n, entry=0.3)
    sparse_matrix_symb_hermitian_block_encoding = SparseMatrixHermitian(
        col_oracle, entry_oracle, eps=0
    )
    return sparse_matrix_symb_hermitian_block_encoding


_SPARSE_MATRIX_HERMITIAN_DOC = BloqDocSpec(
    bloq_cls=SparseMatrixHermitian,
    examples=[
        _sparse_matrix_symb_hermitian_block_encoding,
        _sparse_matrix_hermitian_block_encoding,
    ],
)
