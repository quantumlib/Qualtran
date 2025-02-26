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

from qualtran import Bloq, BloqDocSpec, BQUInt, QDType, Register, Signature
from qualtran.bloqs.arithmetic import Xor
from qualtran.bloqs.arithmetic.sorting import Comparator
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator
from qualtran.symbolics import ceil, log2, SymbolicInt


@frozen
class SortInPlace(Bloq):
    r"""Sort a list of $\ell$ numbers in place using $\ell \log \ell$ ancilla bits.

    Applies the map:
    $$
        |x_1, x_2, \ldots, x_l\rangle
        |0^{\ell \log \ell}\rangle
        \mapsto
        |x_{\pi_1}, x_{\pi_2}, \ldots, x_{\pi_\ell})\rangle
        |\pi_1, \pi_2, \ldots, \pi_\ell\rangle
    $$
    where $x_{\pi_1} \le x_{\pi_2} \ldots \le x_{\pi_\ell}$ is the sorted list,
    and the ancilla are entangled.

    To apply this, we first use any sorting algorithm to output the sorted list
    in a clean register. And then use the following algorithm from Lemma 4.12 of Ref [1]
    that applies the map:

    $$
        |x_1, ..., x_l\rangle|x_{\pi(1)}, ..., x_{\pi(l)})\rangle
        \mapsto
        |x_l, ..., x_l\rangle|\pi(1), ..., \pi(l))\rangle
    $$

    where $x_i \in [n]$ and $\pi(i) \in [l]$.
    This second algorithm (Lemma 4.12) has two steps, each with $l^2$ comparisons:
    1. compute `pi(1) ... pi(l)` given `x_1 ... x_l` and `x_{pi(1)} ... x{pi(l)}`.
    1. (un)compute `x_{pi(1)} ... x{pi(l)}` using `pi(1) ... pi(l)` given `x_1 ... x_l`.

    Args:
        l: number of elements in the list
        dtype: type of each element to store `[n]`.

    Registers:
        input: the entire input as a single register
        ancilla (RIGHT): the generated (entangled) register storing `pi`.

    References:
        [Quartic quantum speedups for planted inference](https://arxiv.org/abs/2406.19378v1)
        Lemma 4.12. Eq. 122.
    """

    l: SymbolicInt
    dtype: QDType

    @property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('xs', self.dtype, shape=(self.l,)),
                Register('pi', self.index_dtype, shape=(self.l,)),
            ]
        )

    @property
    def index_dtype(self) -> QDType:
        """dtype to represent an index in range `[l]`"""
        bitsize = ceil(log2(self.l))
        return BQUInt(bitsize, self.l)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> BloqCountDictT:
        compare = Comparator(self.dtype.num_qubits)
        n_ops = 3 * self.l**2

        counts = Counter[Bloq]()

        counts[compare] += n_ops
        counts[compare.adjoint()] += n_ops
        counts[Xor(self.dtype)] += n_ops

        return counts


_SORT_IN_PLACE_DOC = BloqDocSpec(bloq_cls=SortInPlace, examples=[])
