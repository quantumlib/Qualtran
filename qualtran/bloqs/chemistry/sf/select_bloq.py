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

from functools import cached_property
from typing import Set, TYPE_CHECKING

from attrs import frozen

from qualtran import Bloq, bloq_example, Signature
from qualtran.bloqs.basic_gates import Toffoli

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@frozen
class SelectSingleFactorization(Bloq):
    r"""Single Factorization SELECT bloq.

    Implements selected Majorana Fermion operation.

    Args:
        num_spin_orb: The number of spin orbitals. Typically called N.
        additional_control: whether to control on $l \ne zero$ or not.

    Registers:
        p: spatial orbital index. range(0, num_spin_orb // 2)
        q: spatial orbital index. range(0, num_spin_orb // 2)
        spin: spin index.
        succ_pq: flag for success of this state preparation.
        succ_l: flag for success of l state preparation.

    Refererences:
        [Even More Efficient Quantum Computations of Chemistry Through Tensor
            Hypercontraction](https://arxiv.org/abs/2011.03494) Appendix B, page 44, listing 4.
    """

    num_spin_orb: int
    additional_control: bool = False

    @cached_property
    def signature(self) -> Signature:
        n = (self.num_spin_orb // 2 - 1).bit_length()
        return Signature.build(p=n, q=n, spin=1, succ_pq=1, succ_l=1)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {(Toffoli(), 2 * (self.num_spin_orb - 2))}


@bloq_example
def _select() -> SelectSingleFactorization:
    return SelectSingleFactorization(10)
