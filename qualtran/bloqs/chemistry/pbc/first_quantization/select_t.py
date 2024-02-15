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
r"""Bloqs for SELECT T for the first quantized chemistry Hamiltonian."""
from functools import cached_property
from typing import Set, TYPE_CHECKING

from attrs import frozen

from qualtran import Bloq, bloq_example, BloqDocSpec, Register, Signature
from qualtran.bloqs.basic_gates import Toffoli

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@frozen
class SelectTFirstQuantization(Bloq):
    r"""SELECT for the kinetic energy operator for the first quantized chemistry Hamiltonian.

    Args:
        num_bits_p: The number of bits to represent each dimension of the momentum register.
        eta: The number of electrons.

    Registers:
        sys: The system register.
        plus: A $|+\rangle$ state.
        flag_T: a flag to control on the success of the $T$ state preparation.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](
            https://arxiv.org/abs/2105.12767) page 20, section B
    """
    num_bits_p: int
    eta: int

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("flag_T", bitsize=1),
                Register("plus", bitsize=1),
                Register("w", bitsize=3),
                Register("r", bitsize=self.num_bits_p),
                Register("s", bitsize=self.num_bits_p),
                Register("p", bitsize=self.num_bits_p, shape=(3,)),
            ]
        )

    def short_name(self) -> str:
        return r'SEL $T$'

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # Cost is $5(n_{p} - 1) + 2$ which comes from copying each $w$ component of $p$
        # into an ancilla register ($3(n_{p}-1)$), copying the $r$ and $s$ bit of into an
        # ancilla ($2(n_{p}-1)$), controlling on both those bit perform phase flip on an
        # ancilla $|+\rangle$ state. This requires $1$ Toffoli, Then erase which costs
        # only Cliffords. There is an additional control bit controlling the application
        # of $T$ thus we come to our total.
        # Eq 73. page
        return {(Toffoli(), (5 * (self.num_bits_p - 1) + 2))}


@bloq_example
def _select_t() -> SelectTFirstQuantization:
    num_bits_p = 5
    eta = 10

    select_t = SelectTFirstQuantization(num_bits_p=num_bits_p, eta=eta)
    return select_t


_SELECT_T = BloqDocSpec(
    bloq_cls=SelectTFirstQuantization,
    import_line='from qualtran.bloqs.chemistry.pbc.first_quantization.select_t import SelectTFirstQuantization',
    examples=(_select_t,),
)
