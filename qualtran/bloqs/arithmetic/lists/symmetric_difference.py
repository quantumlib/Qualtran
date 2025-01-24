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
from collections import Counter

from attrs import frozen

from qualtran import Bloq, bloq_example, BloqDocSpec, QBit, QDType, QUInt, Register, Signature
from qualtran.bloqs.arithmetic import Equals, EqualsAConstant, HammingWeightCompute, Xor
from qualtran.bloqs.arithmetic.sorting import BitonicMerge
from qualtran.bloqs.basic_gates import CNOT
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.symbolics import bit_length, is_symbolic, SymbolicInt


@frozen
class SymmetricDifference(Bloq):
    r"""Given two sorted sets $S, T$ of unique elements, compute their symmetric difference.

    This accepts an integer `n_diff`, and marks a flag qubit if the symmetric difference
    set is of size exactly `n_diff`. If the flag is marked (1), then the output of `n_diff`
    numbers is the symmetric difference, otherwise it may be arbitrary.

    Args:
        n_lhs: number of elements in $S$
        n_rhs: number of elements in $T$
        n_diff: expected number of elements in the difference $S \Delta T$.
        dtype: type of each element.

    Registers:
        S: list of `n_lhs` numbers.
        T: list of `n_rhs` numbers.
        diff: output register of `n_diff` numbers.
        flag: 1 if there are duplicates, 0 if all are unique.

    References:
        [Quartic quantum speedups for planted inference](https://arxiv.org/abs/2406.19378v1)
        Theorem 4.17, proof para 3, page 38.
    """

    n_lhs: SymbolicInt
    n_rhs: SymbolicInt
    n_diff: SymbolicInt
    dtype: QDType

    def __attrs_post_init__(self):
        if not is_symbolic(self.n_lhs, self.n_rhs):
            assert self.n_lhs >= self.n_rhs, "lhs must be the larger set"

    @property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('S', self.dtype, shape=(self.n_lhs,)),
                Register('T', self.dtype, shape=(self.n_rhs,)),
                Register('diff', self.dtype, shape=(self.n_diff,)),
                Register('flag', QBit()),
            ]
        )

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> BloqCountDictT:
        # the forward pass, i.e. all bloqs that must be uncomputed
        counts_forward = Counter[Bloq]()

        # merge the lists
        counts_forward[BitonicMerge(self.n_lhs, self.dtype.num_qubits)] += 1
        # compare adjacents
        counts_forward[Equals(self.dtype)] += self.n_lhs + self.n_rhs - 1
        # compute number of equal adjacents
        counts_forward[HammingWeightCompute(self.n_lhs + self.n_rhs - 1)] += 1
        # check: 2 * n_equal = n_lhs + n_rhs - n_diff
        # (note: the above eq holds as we assume all input elements are unique)
        counts_forward[
            EqualsAConstant(
                bit_length(self.n_lhs + self.n_rhs - 1),
                (self.n_lhs + self.n_rhs - self.n_diff) // 2,
            )
        ] += 1

        # all bloqs
        counts = Counter[Bloq]()

        # copy the first n_diff numbers and flag
        counts[Xor(self.dtype)] += self.n_diff
        counts[CNOT()] += 1

        for bloq, n in counts_forward.items():
            counts[bloq] += n
            counts[bloq.adjoint()] += n

        return counts


@bloq_example
def _symm_diff() -> SymmetricDifference:
    dtype = QUInt(4)
    symm_diff = SymmetricDifference(n_lhs=4, n_rhs=2, n_diff=4, dtype=dtype)
    return symm_diff


@bloq_example
def _symm_diff_symb() -> SymmetricDifference:
    import sympy

    from qualtran.symbolics import bit_length

    n, k, c = sympy.symbols("n k c", positive=True, integer=True)
    dtype = QUInt(bit_length(n - 1))
    symm_diff_symb = SymmetricDifference(n_lhs=c * k, n_rhs=k, n_diff=c * k, dtype=dtype)
    return symm_diff_symb


@bloq_example
def _symm_diff_equal_size_symb() -> SymmetricDifference:
    import sympy

    from qualtran.symbolics import bit_length

    n, k, c = sympy.symbols("n k c", positive=True, integer=True)
    dtype = QUInt(bit_length(n - 1))
    symm_diff_equal_size_symb = SymmetricDifference(n_lhs=c * k, n_rhs=c * k, n_diff=k, dtype=dtype)
    return symm_diff_equal_size_symb


_SYMMETRIC_DIFFERENCE_DOC = BloqDocSpec(bloq_cls=SymmetricDifference, examples=[_symm_diff])
