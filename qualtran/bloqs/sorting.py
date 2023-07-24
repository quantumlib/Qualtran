#  Copyright 2023 Google LLC
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

import numpy as np
from attrs import frozen
from cirq_ft import TComplexity

from qualtran import Bloq, Register, Side, Signature
from qualtran.bloqs.arithmetic import GreaterThan


@frozen
class Comparator(Bloq):
    r"""Compare and potentially swaps two n-bit numbers.

    Implements $U|a\rangle|b\rangle|0\rangle \rightarrow |\min(a,b)\rangle|\max(a,b)\rangle|a>b\rangle$,

    where $a$ and $b$ are n-qubit quantum registers. On output a and b are
    swapped if a > b. Forms the base primitive for sorting.

    Args:
        bitsize: Number of bits used to represent each integer.

    Registers:
     - a: A nbit-sized input register (register a above).
     - b: A nbit-sized input register (register b above).
     - out: A single bit output register which will store the result of the comparator.

    References:
        [Improved techniques for preparing eigenstates of fermionic
        Hamiltonians](https://www.nature.com/articles/s41534-018-0071-5),
        Fig. 1. in main text.
    """

    bitsize: int

    @property
    def signature(self):
        return Signature(
            [
                Register('a', 1, shape=(self.bitsize,)),
                Register('b', 1, shape=(self.bitsize,)),
                Register('out', 1, side=Side.RIGHT),
            ]
        )

    def short_name(self) -> str:
        return "Cmprtr"

    def t_complexity(self):
        # complexity is from less than on two n qubit numbers + controlled swap
        # Hard code for now until CSwap-Bloq is merged.
        # See: https://github.com/quantumlib/cirq-qubitization/issues/219
        t_complexity = GreaterThan(self.bitsize).t_complexity()
        t_complexity += TComplexity(t=14 * self.bitsize)
        return t_complexity


@frozen
class BitonicSort(Bloq):
    r"""Sort k n-bit numbers.

    TODO: actually implement the algorithm using comparitor. Hiding ancilla cost
        for the moment. Issue #219

    Args:
        bitsize: Number of bits used to represent each integer.
        k: Number of integers to sort.

    Registers:
     - input: A k-nbit-sized input register (register a above). List of integers
        we want to sort.

    References:
        [Improved techniques for preparing eigenstates of fermionic
        Hamiltonians](https://www.nature.com/articles/s41534-018-0071-5),
        Supporting Information Sec. II.
    """

    bitsize: int
    k: int

    @property
    def signature(self):
        return Signature([Register("input", bitsize=self.bitsize, shape=(self.bitsize,))])

    def short_name(self) -> str:
        return "BSort"

    def t_complexity(self):
        # Need O(k * log^2(k)) comparisons.
        # TODO: This is Big-O complexity.
        # Should work out constant factors or
        # See: https://github.com/quantumlib/cirq-qubitization/issues/219
        return (
            self.k
            * int(np.ceil(max(np.log2(self.k) ** 2.0, 1)))
            * Comparator(self.bitsize).t_complexity()
        )
