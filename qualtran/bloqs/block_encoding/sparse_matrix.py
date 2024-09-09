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
from typing import Dict, Iterable, Tuple

import numpy as np
import sympy
from attrs import field, frozen
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    BQUInt,
    DecomposeTypeError,
    QAny,
    QBit,
    QUInt,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.arithmetic import Add, AddK
from qualtran.bloqs.basic_gates import Ry, Swap
from qualtran.bloqs.block_encoding import BlockEncoding
from qualtran.bloqs.bookkeeping.auto_partition import AutoPartition, Unused
from qualtran.bloqs.data_loading import QROM
from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity
from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare
from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
    PrepareUniformSuperposition,
)
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.simulation.classical_sim import ClassicalValT
from qualtran.symbolics import is_symbolic, SymbolicFloat, SymbolicInt
from qualtran.symbolics.math_funcs import bit_length


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
        if not is_symbolic(self.system_bitsize) and self.system_bitsize <= 0:
            raise ValueError("system_bitsize must be > 0")
        if is_symbolic(self.num_nonzero):
            return
        if self.num_nonzero <= 0:
            raise ValueError("num_nonzero must be > 0")
        if self.num_nonzero > 2**self.system_bitsize:
            raise ValueError("Cannot have more than 2 ** system_bitsize non-zero elements")

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(
            l=BQUInt(self.system_bitsize, self.num_nonzero), i=QUInt(self.system_bitsize)
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
    def signal_state(self) -> BlackBoxPrepare:
        return BlackBoxPrepare(PrepareIdentity.from_bitsizes([self.ancilla_bitsize]))

    @cached_property
    def diffusion(self):
        unused = self.system_bitsize - bit_length(self.row_oracle.num_nonzero - 1)
        return AutoPartition(
            PrepareUniformSuperposition(n=self.row_oracle.num_nonzero),
            partitions=[
                (Register("target", QAny(self.system_bitsize)), [Unused(unused), "target"])
            ],
        )

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:
        return {
            self.diffusion: 1,
            self.col_oracle: 1,
            self.entry_oracle: 1,
            Swap(self.system_bitsize): 1,
            self.row_oracle.adjoint(): 1,
            self.diffusion.adjoint(): 1,
        }

    def build_composite_bloq(
        self, bb: BloqBuilder, system: SoquetT, ancilla: SoquetT
    ) -> Dict[str, SoquetT]:
        if is_symbolic(self.system_bitsize) or is_symbolic(self.row_oracle.num_nonzero):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self=}")

        assert not isinstance(ancilla, np.ndarray)
        ancilla_bits = bb.split(ancilla)
        q, l = ancilla_bits[0], bb.join(ancilla_bits[1:])

        l = bb.add(self.diffusion, target=l)
        l, system = bb.add(self.col_oracle, l=l, i=system)
        q, l, system = bb.add(self.entry_oracle, q=q, i=l, j=system)
        l, system = bb.add(Swap(self.system_bitsize), x=l, y=system)
        l, system = bb.add(self.row_oracle.adjoint(), l=l, i=system)
        l = bb.add(self.diffusion.adjoint(), target=l)

        return {"system": system, "ancilla": bb.join(np.concatenate([[q], bb.split(l)]))}

    def __str__(self) -> str:
        return "B[SparseMatrix]"


@frozen
class TopLeftRowColumnOracle(RowColumnOracle):
    """Oracle specifying the non-zero rows or columns of a matrix with top left block non-zero.

    Args:
        system_bitsize: The number of bits used to represent an index.
        num_nonzero: The number of rows or columns of the non-zero top left block. If unspecified,
            defaults to making the entire matrix non-zero.

    Registers:
        l: As input, index specifying the `l`-th non-zero entry to find in row / column `i`.
           As output, position of the `l`-th non-zero entry in row / column `i`.
        i: The row / column index.
    """

    system_bitsize: SymbolicInt
    _num_nonzero: SymbolicInt = field()

    @_num_nonzero.default
    def _num_nonzero_default(self):
        # Default to a matrix with all entries non-zero
        return 2**self.system_bitsize

    @cached_property
    def num_nonzero(self) -> SymbolicInt:
        return self._num_nonzero

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> Dict[str, SoquetT]:
        # the l-th non-zero entry is at position l, so do nothing
        return soqs


@frozen
class SymmetricBandedRowColumnOracle(RowColumnOracle):
    """Oracle specifying the non-zero rows and columns of a symmetric
    [banded matrix](https://en.wikipedia.org/wiki/Band_matrix).

    The symmetry here refers to the pattern of non-zero entries, not necessarily the entries themselves, which are determined separately by the `EntryOracle`.

    Args:
        system_bitsize: The number of bits used to represent an index.
        bandsize: The number of pairs of non-zero off-main diagonals. A diagonal matrix has
            bandsize 0 and a tridiagonal matrix has bandsize 1.

    Registers:
        l: As input, index specifying the `l`-th non-zero entry to find in row / column `i`.
           As output, position of the `l`-th non-zero entry in row / column `i`.
        i: The row / column index.
    """

    system_bitsize: SymbolicInt
    bandsize: SymbolicInt

    @cached_property
    def num_nonzero(self) -> SymbolicInt:
        return 2 * self.bandsize + 1

    def __attrs_post_init__(self):
        if is_symbolic(self.system_bitsize) or is_symbolic(self.bandsize):
            return
        if 2**self.system_bitsize < 2 * self.bandsize:
            raise ValueError(
                f"bandsize={self.bandsize} too large for system_bitsize={self.system_bitsize}"
            )

    def call_classically(self, l: ClassicalValT, i: ClassicalValT) -> Tuple[ClassicalValT, ...]:
        if (
            is_symbolic(self.bandsize)
            or is_symbolic(self.system_bitsize)
            or is_symbolic(self.num_nonzero)
        ):
            raise DecomposeTypeError(f"Cannot call symbolic {self=} classically")

        assert not isinstance(l, np.ndarray) and not isinstance(i, np.ndarray)
        if l >= self.num_nonzero:
            raise IndexError("l out of bounds")
        return ((l + i - self.bandsize) % (2**self.system_bitsize), i)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> BloqCountDictT:
        return {
            Add(QUInt(self.system_bitsize), QUInt(self.system_bitsize)): 1,
            AddK(self.system_bitsize, -self.bandsize, signed=True): 1,
        }

    def build_composite_bloq(self, bb: BloqBuilder, l: SoquetT, i: SoquetT) -> Dict[str, SoquetT]:
        if is_symbolic(self.system_bitsize) or is_symbolic(self.bandsize):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self=}")

        i, l = bb.add(Add(QUInt(self.system_bitsize), QUInt(self.system_bitsize)), a=i, b=l)
        l = bb.add(AddK(self.system_bitsize, -self.bandsize, signed=True), x=l)

        return {"l": l, "i": i}


@frozen
class UniformEntryOracle(EntryOracle):
    """Oracle specifying the entries of a matrix with uniform entries."""

    system_bitsize: SymbolicInt
    entry: float

    def build_composite_bloq(
        self, bb: BloqBuilder, q: Soquet, **soqs: SoquetT
    ) -> Dict[str, SoquetT]:
        # Either Rx or Ry work here; Rx would induce a phase on the subspace with non-zero ancilla
        # See https://arxiv.org/abs/2302.10949 for reference that uses Rx
        soqs["q"] = bb.add(Ry(2 * np.arccos(self.entry)), q=q)
        return soqs


@frozen
class ExplicitEntryOracle(EntryOracle):
    """Oracle specifying the entries of a matrix as given by an explicit array.

    Under the hood, implemented via QROM and quantum variable rotation. This bloq is useful for
    prototyping and testing; its cost is larger than that of a truly sparse/structured matrix.

    Args:
        system_bitsize: The number of bits used to represent an index.
        data: 2-D array of matrix entries. All entries must be >= 0 and < 1.
        entry_bitsize: The number of bits of precision to represent the arccos of each entry.

    Registers:
        q: The flag qubit that is rotated proportionally to the value of the entry.
        i: The row index.
        j: The column index.
    """

    system_bitsize: SymbolicInt
    data: NDArray[np.float64] = field(
        converter=lambda x: np.asarray(x) if isinstance(x, Iterable) else x,
        eq=lambda d: tuple(d.flat),
    )
    entry_bitsize: SymbolicInt

    def __attrs_post_init__(self):
        if len(self.data.shape) != 2:
            raise ValueError("data must be a 2-D array")
        x, y = self.data.shape
        if x != y:
            raise ValueError("data must be square")
        if not is_symbolic(self.system_bitsize) and x != 2**self.system_bitsize:
            raise ValueError("data must have dimension 2 ** self.system_bitsize")
        if not is_symbolic(self.entry_bitsize) and self.entry_bitsize < 1:
            raise ValueError("entry_bitsize must be >= 1")
        if not all(x >= 0 and x <= 1 for x in self.data.flat):
            raise ValueError("entries must be >= 0 and <= 1")

    def build_composite_bloq(
        self, bb: BloqBuilder, q: SoquetT, i: SoquetT, j: SoquetT
    ) -> Dict[str, SoquetT]:
        if is_symbolic(self.entry_bitsize):
            raise DecomposeTypeError(f"Cannot decompose symbolic {self=}")
        # load from QROM
        data = np.arccos(self.data) / np.pi * 2**self.entry_bitsize
        qrom = QROM.build_from_data(data, target_bitsizes=(self.entry_bitsize,))
        target = bb.allocate(self.entry_bitsize)
        i, j, target = bb.add(qrom, selection0=i, selection1=j, target0_=target)
        # perform fractional Ry
        # can't use StatePreparationViaRotations here because the coefficients depend on i, j
        # can't use QvrZPow here because CRz is not symmetric and we condition on target, not q
        # TODO: could potentially use RzViaPhaseGradient when it is done
        target_bits = bb.split(target)
        for k in range(len(target_bits)):
            target_bits[k], q = bb.add(
                Ry(2 * np.pi * (2 ** -(k + 1))).controlled(), ctrl=target_bits[k], q=q
            )
        target = bb.join(target_bits)
        # unload from QROM
        i, j, target = bb.add(qrom.adjoint(), selection0=i, selection1=j, target0_=target)
        bb.free(target)
        return {"q": q, "i": i, "j": j}


@bloq_example
def _sparse_matrix_block_encoding() -> SparseMatrix:
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        TopLeftRowColumnOracle,
        UniformEntryOracle,
    )

    row_oracle = TopLeftRowColumnOracle(system_bitsize=2)
    col_oracle = TopLeftRowColumnOracle(system_bitsize=2)
    entry_oracle = UniformEntryOracle(system_bitsize=2, entry=0.3)
    sparse_matrix_block_encoding = SparseMatrix(row_oracle, col_oracle, entry_oracle, eps=0)
    return sparse_matrix_block_encoding


@bloq_example
def _sparse_matrix_symb_block_encoding() -> SparseMatrix:
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        TopLeftRowColumnOracle,
        UniformEntryOracle,
    )

    n = sympy.Symbol('n', positive=True, integer=True)
    row_oracle = TopLeftRowColumnOracle(system_bitsize=n)
    col_oracle = TopLeftRowColumnOracle(system_bitsize=n)
    entry_oracle = UniformEntryOracle(system_bitsize=n, entry=0.3)
    sparse_matrix_symb_block_encoding = SparseMatrix(row_oracle, col_oracle, entry_oracle, eps=0)
    return sparse_matrix_symb_block_encoding


@bloq_example
def _explicit_matrix_block_encoding() -> SparseMatrix:
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        ExplicitEntryOracle,
        TopLeftRowColumnOracle,
    )

    data = np.array([[0.0, 0.25], [1 / 3, 0.467]])
    row_oracle = TopLeftRowColumnOracle(system_bitsize=1)
    col_oracle = TopLeftRowColumnOracle(system_bitsize=1)
    entry_oracle = ExplicitEntryOracle(system_bitsize=1, data=data, entry_bitsize=10)
    explicit_matrix_block_encoding = SparseMatrix(row_oracle, col_oracle, entry_oracle, eps=0)
    return explicit_matrix_block_encoding


@bloq_example
def _symmetric_banded_matrix_block_encoding() -> SparseMatrix:
    from qualtran.bloqs.block_encoding.sparse_matrix import SymmetricBandedRowColumnOracle

    row_oracle = SymmetricBandedRowColumnOracle(3, bandsize=1)
    col_oracle = SymmetricBandedRowColumnOracle(3, bandsize=1)
    entry_oracle = UniformEntryOracle(3, entry=0.3)
    symmetric_banded_matrix_block_encoding = SparseMatrix(
        row_oracle, col_oracle, entry_oracle, eps=0
    )
    return symmetric_banded_matrix_block_encoding


_SPARSE_MATRIX_DOC = BloqDocSpec(
    bloq_cls=SparseMatrix,
    examples=[
        _sparse_matrix_block_encoding,
        _sparse_matrix_symb_block_encoding,
        _explicit_matrix_block_encoding,
        _symmetric_banded_matrix_block_encoding,
    ],
)
