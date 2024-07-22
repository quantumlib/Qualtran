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
from functools import cached_property
from typing import cast, Dict, Tuple

import numpy as np
from attrs import frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    BoundedQUInt,
    QAny,
    QBit,
    QUInt,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.basic_gates import Rx, Swap
from qualtran.bloqs.block_encoding import BlockEncoding
from qualtran.bloqs.block_encoding.lcu_select_and_prepare import PrepareOracle
from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
    PrepareUniformSuperposition,
)
from qualtran.symbolics import SymbolicFloat, SymbolicInt


@frozen
class RowColumnOracle(Bloq, abc.ABC):
    r"""Oracle specifying the non-zero rows or columns of a sparse-access matrix.

    In the reference, this is the interface of
    $O_r : \ket{\ell}\ket{i} \mapsto \ket{r(i, \ell)}\ket{i}$, and of
    $O_c : \ket{\ell}\ket{j} \mapsto \ket{c(j, \ell)}\ket{j}$.
    Here, $r(i, \ell)$ and $c(j, \ell)$ give the $\ell$-th nonzero entry in the $i$-th row
    and $j$-th column of the matrix, respectively.

    Registers:
        l: As input, index specifying the `l`-th non-zero entry to find in row / column `i`.
           As output, position of the `l`-th non-zero entry in row / column `i`.
        i: The row / column index.

    References:
        [Lecture Notes on Quantum Algorithms for Scientific Computation](https://arxiv.org/abs/2201.08309). Lin Lin (2022). Ch. 6.5.
    """

    @property
    @abc.abstractmethod
    def system_bitsize(self) -> SymbolicInt:
        """The number of bits used to represent an index."""

    @property
    @abc.abstractmethod
    def num_nonzero(self) -> SymbolicInt:
        """The number of nonzero entries in each row or column."""

    def __attrs_post_init__(self):
        if self.num_nonzero > 2**self.system_bitsize:
            raise ValueError("Cannot have more than 2 ** system_bitsize non-zero elements")

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(
            l=BoundedQUInt(self.system_bitsize, self.num_nonzero), i=QUInt(self.system_bitsize)
        )


@frozen
class EntryOracle(Bloq, abc.ABC):
    r"""Oracle specifying the entries of a sparse-access matrix.

    In the reference, this is the interface of
    $$O_A : \ket{0}\ket{i}\ket{j} \mapsto (A_{ij}\ket{0} + \sqrt{1 - |A_{ij}|^2}\ket{i}\ket{j}).$$

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

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(
            q=QBit(), i=QAny(self.system_bitsize), j=QAny(self.system_bitsize)
        )


@frozen
class SparseMatrix(BlockEncoding):
    r"""Block encoding of a sparse-access matrix.

    Given row, column, and entry oracles $O_r$, $O_c$, and $O_A$ for an $s$-sparse matrix
    $A \in \mathbb{C}^{2^n \times 2^n}$, i.e. one where each row / column has exactly $s$ non-zero
    entries, computes a $(s, n+1, \epsilon)$-block encoding of $A$ as follows:
    ```
           ┌────┐                       ┌────┐
      |0> ─┤    ├─     |0> ─────────────┤    ├───────────────
           │    │           ┌──┐        │    │          ┌──┐
           │ U  │  =        │ n│ ┌────┐ │ O  │   ┌────┐ │ n│
    |0^n> ─┤  A ├─   |0^n> ─┤H ├─┤    ├─┤  A ├─X─┤    ├─┤H ├─
           │    │           └──┘ │ O  │ │    │ │ │ O* │ └──┘
    |Psi> ─┤    ├─   |Psi> ──────┤  c ├─┤    ├─X─┤  r ├──────
           └────┘                └────┘ └────┘   └────┘
    ```

    To encode a matrix of irregular dimension, the matrix should first be embedded into one of
    dimension $2^n \times 2^n$ for suitable $n$.
    To encode a matrix where each row / column has at most $s$ non-zero entries, some zeroes should
    be treated as if they were non-zero so that each row / column has exactly $s$ non-zero entries.

    Args:
        row_oracle: The row oracle $O_r$. See `RowColumnOracle` for definition.
        col_oracle: The column oracle $O_c$. See `RowColumnOracle` for definition.
        entry_oracle: The entry oracle $O_A$. See `EntryOracle` for definition.
        eps: The precision of the block encoding.

    Registers:
        system: The system register.
        ancilla: The ancilla register.
        resource: The resource register (present only if bitsize > 0).

    References:
        [Lecture Notes on Quantum Algorithms for Scientific Computation](https://arxiv.org/abs/2201.08309). Lin Lin (2022). Ch. 6.5.
    """

    row_oracle: RowColumnOracle
    col_oracle: RowColumnOracle
    entry_oracle: EntryOracle
    eps: SymbolicFloat

    def __attrs_post_init__(self):
        if (
            self.row_oracle.system_bitsize != self.col_oracle.system_bitsize
            or self.row_oracle.system_bitsize != self.entry_oracle.system_bitsize
        ):
            raise ValueError("Row, column, and entry oracles must have same bitsize")
        if self.row_oracle.num_nonzero != self.col_oracle.num_nonzero:
            raise ValueError("Unequal row and column sparsities are not supported")

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(
            system=QAny(self.system_bitsize),
            ancilla=QAny(self.ancilla_bitsize),
            resource=QAny(self.resource_bitsize),  # if resource_bitsize is 0, not present
        )

    @cached_property
    def system_bitsize(self) -> SymbolicInt:
        return self.entry_oracle.system_bitsize

    def pretty_name(self) -> str:
        return "B[SparseMatrix]"

    @cached_property
    def alpha(self) -> SymbolicFloat:
        return self.row_oracle.num_nonzero

    @cached_property
    def ancilla_bitsize(self) -> SymbolicInt:
        return self.system_bitsize + 1

    @cached_property
    def resource_bitsize(self) -> SymbolicInt:
        return 0

    @cached_property
    def epsilon(self) -> SymbolicFloat:
        return self.eps

    @property
    def target_registers(self) -> Tuple[Register, ...]:
        return (self.signature.get_right("system"),)

    @property
    def junk_registers(self) -> Tuple[Register, ...]:
        return (self.signature.get_right("resource"),)

    @property
    def selection_registers(self) -> Tuple[Register, ...]:
        return (self.signature.get_right("ancilla"),)

    @property
    def signal_state(self) -> PrepareOracle:
        # This method will be implemented in the future after PrepareOracle
        # is updated for the BlockEncoding interface.
        # Github issue: https://github.com/quantumlib/Qualtran/issues/1104
        raise NotImplementedError

    def build_composite_bloq(
        self, bb: BloqBuilder, system: SoquetT, ancilla: SoquetT
    ) -> Dict[str, SoquetT]:
        ancilla_bits = bb.split(cast(Soquet, ancilla))
        q, l = ancilla_bits[0], bb.join(ancilla_bits[1:])

        diffusion = PrepareUniformSuperposition(n=2**self.system_bitsize)
        l = bb.add(diffusion, target=l)
        l, system = bb.add_t(self.col_oracle, l=cast(Soquet, l), i=system)
        q, l, system = bb.add_t(self.entry_oracle, q=q, i=l, j=system)
        l, system = bb.add_t(Swap(self.system_bitsize), x=l, y=system)
        l, system = bb.add_t(self.row_oracle.adjoint(), l=l, i=system)
        l = bb.add(diffusion.adjoint(), target=l)

        return {"system": system, "ancilla": bb.join(np.concatenate([[q], bb.split(l)]))}


@frozen
class FullRowColumnOracle(RowColumnOracle):
    """Oracle specifying the non-zero rows or columns of a matrix with full entries."""

    system_bitsize: SymbolicInt

    @cached_property
    def num_nonzero(self) -> SymbolicInt:
        return 2**self.system_bitsize

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        # the l-th non-zero entry is at position l, so do nothing
        return soqs


@frozen
class UniformEntryOracle(EntryOracle):
    """Oracle specifying the entries of a matrix with uniform entries."""

    system_bitsize: SymbolicInt
    entry: float

    def build_composite_bloq(
        self, bb: BloqBuilder, q: Soquet, **soqs: SoquetT
    ) -> Dict[str, SoquetT]:
        soqs["q"] = cast(Soquet, bb.add(Rx(2 * np.arccos(self.entry)), q=q))
        return soqs


@bloq_example
def _sparse_matrix_block_encoding() -> SparseMatrix:
    from qualtran.bloqs.block_encoding.sparse_matrix import FullRowColumnOracle, UniformEntryOracle

    row_oracle = FullRowColumnOracle(2)
    col_oracle = FullRowColumnOracle(2)
    entry_oracle = UniformEntryOracle(system_bitsize=2, entry=0.3)
    sparse_matrix_block_encoding = SparseMatrix(row_oracle, col_oracle, entry_oracle, eps=0)
    return sparse_matrix_block_encoding


_SPARSE_MATRIX_DOC = BloqDocSpec(
    bloq_cls=SparseMatrix,
    import_line="from qualtran.bloqs.block_encoding import SparseMatrix",
    examples=[_sparse_matrix_block_encoding],
)
