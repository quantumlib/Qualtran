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

from qualtran import Bloq, BloqDocSpec, QAny, QBit, QDType, Signature
from qualtran.bloqs.arithmetic import LessThanEqual
from qualtran.bloqs.mcmt import MultiControlX
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.symbolics import HasLength, SymbolicInt


@frozen
class HasDuplicates(Bloq):
    r"""Given a sorted list of `l` numbers, check if it contains any duplicates.

    Produces a single qubit which is `1` if there are duplicates, and `0` if all are disjoint.
    It compares every adjacent pair, and therefore uses `l - 1` comparisons.
    It then uses a single MCX on `l - 1` bits gate to compute the flag.

    Args:
        l: number of elements in the list
        dtype: type of each element to store `[n]`.

    Registers:
        input: the entire input as a single register
        flag (RIGHT): 1 if there are duplicates, 0 if all are unique.

    References:
        [Quartic quantum speedups for planted inference](https://arxiv.org/abs/2406.19378v1)
        Lemma 4.12. Eq. 122.
    """

    l: SymbolicInt
    dtype: QDType

    @property
    def signature(self) -> 'Signature':
        return Signature.build_from_dtypes(input=QAny(self.input_bitsize), flag=QBit())

    @property
    def input_bitsize(self) -> SymbolicInt:
        return self.l * self.dtype.num_qubits

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> BloqCountDictT:
        logn = self.dtype.num_qubits

        counts = Counter[Bloq]()

        counts[LessThanEqual(logn, logn)] += self.l - 1
        counts[MultiControlX(cvs=HasLength(self.l - 1))] += 1
        counts[LessThanEqual(logn, logn).adjoint()] += self.l - 1

        return counts


_HAS_DUPLICATES_DOC = BloqDocSpec(bloq_cls=HasDuplicates, examples=[])
