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
r"""Bloqs for SELECT for the U and V parts of the first quantized chemistry Hamiltonian."""
from collections import Counter
from functools import cached_property
from typing import TYPE_CHECKING

from attrs import frozen

from qualtran import Bloq, bloq_example, QAny, QBit, QInt, Register, Signature
from qualtran.bloqs.arithmetic import Add, SignedIntegerToTwosComplement
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.bloqs.chemistry.pbc.first_quantization.select_uv import ApplyNuclearPhase

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, BloqCountT, SympySymbolAllocator


@frozen
class SelectUVFirstQuantizationWithProj(Bloq):
    r"""SELECT for the coulomb operators for the first quantized chemistry Hamiltonian.

    Here we include a quantum projectile.

    This does not include the controlled swaps from p_i and q_j system registers
    into ancilla registers and back again. Hence there is no system register.

    Args:
        num_bits_p: The number of bits to represent each dimension of the momentum register.
        num_bits_n: The number of bits to represent each dimension of the
            momentum register for the projectile.
        eta: The number of electrons.
        num_atoms: The number of atoms.
        num_bits_nuc_pos: The number of bits to store each component of the
            nuclear positions. $n_R$ in the reference.

    Registers:

    References:
        [Quantum computation of stopping power for inertial fusion target design](https://arxiv.org/abs/2308.12352)
        page 11, C also page 34 App A. Sec 3.

        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](https://arxiv.org/abs/2105.12767)
    """

    num_bits_p: int
    num_bits_n: int
    eta: int
    num_atoms: int
    num_bits_nuc_pos: int

    @cached_property
    def signature(self) -> Signature:
        n_nu = self.num_bits_n + 1
        return Signature(
            [
                Register("flag_tuv", QBit()),
                Register("flag_uv", QBit()),
                Register("l", QAny(bitsize=(self.num_atoms - 1).bit_length())),
                Register("rl", QAny(bitsize=self.num_bits_nuc_pos)),
                Register("nu", QAny(bitsize=n_nu), shape=(3,)),
                Register("p", QAny(bitsize=self.num_bits_n), shape=(3,)),
                Register("q", QAny(bitsize=self.num_bits_p), shape=(3,)),
            ]
        )

    def pretty_name(self) -> str:
        return r'SEL UV'

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        cost = Counter['Bloq']()
        # tc_p and tc_n
        # C8 and C9, one of the registers of size num_bits_n so need to account for this.
        cost[SignedIntegerToTwosComplement(self.num_bits_p)] += 3
        cost[SignedIntegerToTwosComplement(self.num_bits_n)] += 3
        # Adding nu into p / q. Nu is one bit larger than p.
        cost[Add(QInt(self.num_bits_p + 1))] += 3
        cost[Add(QInt(self.num_bits_n + 1))] += 3
        # ctrl_add_p and ctrl_add_n
        cost[Toffoli()] += 3 * (self.num_bits_p + 1)
        cost[Toffoli()] += 3 * (self.num_bits_n + 1)
        # inv_tc_p and inv_tc_n
        # + 2 as these numbers are larger from addition of $\nu$
        cost[SignedIntegerToTwosComplement(self.num_bits_p + 2)] += 3
        cost[SignedIntegerToTwosComplement(self.num_bits_n + 2)] += 3
        # cost for phase:
        # 2. Phase by $e^{ik\cdot R}$ in the case of $U$ only.
        cost[ApplyNuclearPhase(self.num_bits_n, self.num_bits_nuc_pos)] += 1
        return cost


@bloq_example
def _sel_uv_proj() -> SelectUVFirstQuantizationWithProj:
    num_bits_n = 8
    num_bits_p = 6
    num_atoms = 8
    eta = 32
    num_bits_nuc_pos = 32
    sel_uv_proj = SelectUVFirstQuantizationWithProj(
        num_bits_p, num_bits_n, eta, num_atoms, num_bits_nuc_pos
    )
    return sel_uv_proj
