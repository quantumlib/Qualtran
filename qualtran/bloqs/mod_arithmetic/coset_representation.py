#  Copyright 2025 Google LLC
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
from typing import TYPE_CHECKING

import attrs
import sympy

from qualtran import Bloq, bloq_example, BloqDocSpec, QUInt, Register, Side, Signature
from qualtran.bloqs.basic_gates import CNOT, Hadamard
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator

if TYPE_CHECKING:
    from qualtran.symbolics import SymbolicInt


@attrs.frozen
class InitCosetRepresntation(Bloq):
    r"""A state initialization of an integer in the coset representation.

    The cost representation of an integer $k$ modulo $N$ with $c_{pad}$ bits is defined as
    $$
        \frac{1}{\sqrt{2^{c_{pad}}}}\sum_{j=0}^{2^{c_{pad}}} \ket{jN + k}
    $$

    This bloq can be built of only clifford gates ... namely $c_{pad}$ `H` gates on the padding
    qubitsfollowed by `CNOT` gates implementing the reversible operation $jN+k$.

    Args:
        c_pad: The number of padding bits.
        k_bitsize: The number of bits used to represent $k$ ($\geq$ the number of bits of $k$ and $N$).
        k: The value of $k$.
        mod: The value of $N$.

    Registers:
        x: A k_bitsize+c_pad register output register containing the initialized state.

    References:
        - [Shor's algorithm with fewer (pure) qubits](https://arxiv.org/abs/quant-ph/0601097)
            section 4.
        - [How to factor 2048 bit RSA integers in 8 hours using 20 million noisy qubits](https://arxiv.org/abs/1905.09749)
            section 2.4
    """

    c_pad: 'SymbolicInt'
    k_bitsize: 'SymbolicInt'
    k: 'SymbolicInt'
    mod: 'SymbolicInt'

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', QUInt(self.c_pad + self.k_bitsize), side=Side.RIGHT)])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        bitsize = self.k_bitsize + self.c_pad
        return {
            Hadamard(): self.c_pad,
            # The matrix representing the reversible operation $jN+x$ consists of 0s and 1s.
            # Thus it can be implemented using CNOTs using LUP decomposition or Gaussian elemenation
            # utilizing at most $n(n-1)$ CNOTs.
            CNOT(): bitsize * (bitsize - 1),
        }


@bloq_example
def _init_coset_representation() -> InitCosetRepresntation:
    c_pad, k_bitsize = sympy.symbols('c k')
    init_coset_representation = InitCosetRepresntation(c_pad, k_bitsize, k=1, mod=19)
    return init_coset_representation


_INIT_COST_REPRESENTATION_DOC = BloqDocSpec(
    bloq_cls=InitCosetRepresntation, examples=[_init_coset_representation]
)
