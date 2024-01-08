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
from functools import cached_property
from typing import Set, TYPE_CHECKING

from attrs import frozen

from qualtran import Bloq, bloq_example, BloqDocSpec, Register, Signature
from qualtran.bloqs.arithmetic import Add, SignedIntegerToTwosComplement
from qualtran.bloqs.basic_gates import Toffoli

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@frozen
class ApplyNuclearPhase(Bloq):
    r"""Apply the phase factor $-e^{-ik_\nu\cdot R_\ell}$ to the state.

    Args:
        num_bits_p: Number of bits for the momentum register.
        num_bits_nuc: Number of bits of precision for the nuclear positions.

    Registers:
        l: A register indexing the nuclear positions.
        rl: A register storing the value of $R_\ell$.
        nu: The momentum transfer register.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](
            https://arxiv.org/abs/2105.12767) pg 25, paragraph 2.
    """

    num_bits_p: int
    num_bits_nuc: int

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("l", bitsize=self.num_bits_nuc),
                Register("Rl", bitsize=self.num_bits_nuc),
                Register("nu", bitsize=self.num_bits_p, shape=(3,)),
            ]
        )

    def short_name(self) -> str:
        return r'$-e^{-k_\nu\cdot R_l$'

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        n_p = self.num_bits_p
        n_n = self.num_bits_nuc
        # This is some complicated application of phase gradient gates.
        # Eq. 97.
        if n_n > n_p:
            cost = 3 * (2 * n_p * n_n - n_p * (n_p + 1) - 1)
        else:
            cost = 3 * n_n * (n_n - 1)
        return {(Toffoli(), cost)}


@frozen
class SelectUVFirstQuantization(Bloq):
    r"""SELECT for the U and V operators for the first quantized chemistry Hamiltonian.

    This does not include the controlled swaps from p_i and q_j system registers
    into ancilla registers and back again. Hence there is no system register.

    Args:
        num_bits_p: The number of bits to represent each dimension of the momentum register.
        eta: The number of electrons.
        num_atoms: The number of atoms.
        num_bits_nuc_pos: The number of bits to store each component of the
            nuclear positions. $n_R$ in the reference.

    Registers:

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](
            https://arxiv.org/abs/2105.12767)
    """

    num_bits_p: int
    eta: int
    num_atoms: int
    num_bits_nuc_pos: int

    @cached_property
    def signature(self) -> Signature:
        n_nu = self.num_bits_p + 1
        return Signature(
            [
                Register("flag_tuv", bitsize=1),
                Register("flag_uv", bitsize=1),
                Register("l", bitsize=(self.num_atoms - 1).bit_length()),
                Register("rl", bitsize=self.num_bits_nuc_pos),
                Register("nu", bitsize=n_nu, shape=(3,)),
                Register("p", bitsize=self.num_bits_p, shape=(3,)),
                Register("q", bitsize=self.num_bits_p, shape=(3,)),
            ]
        )

    def short_name(self) -> str:
        return r'SEL UV'

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        cost_tc = (SignedIntegerToTwosComplement(self.num_bits_p), 6)
        cost_add = (Add(self.num_bits_p + 1), 6)  # + 2?
        cost_ctrl_add = (Toffoli(), 6 * (self.num_bits_p + 1))
        # + 2 as these numbers are larger from addition of $\nu$
        cost_inv_tc = (SignedIntegerToTwosComplement(self.num_bits_p + 2), 6)
        # 2. Phase by $e^{ik\cdot R}$ in the case of $U$ only.
        cost_phase = (ApplyNuclearPhase(self.num_bits_p, self.num_bits_nuc_pos), 1)
        return {cost_tc, cost_add, cost_ctrl_add, cost_inv_tc, cost_phase}


@bloq_example
def _select_uv() -> SelectUVFirstQuantization:
    num_bits_p = 5
    eta = 10
    num_bits_nuc_pos = 16

    select_uv = SelectUVFirstQuantization(
        num_bits_p=num_bits_p, eta=eta, num_atoms=eta, num_bits_nuc_pos=num_bits_nuc_pos
    )
    return select_uv


_SELECT_UV = BloqDocSpec(
    bloq_cls=SelectUVFirstQuantization,
    import_line='from qualtran.bloqs.chemistry.pbc.first_quantization.select_uv import SelectUVFirstQuantization',
    examples=(_select_uv,),
)
