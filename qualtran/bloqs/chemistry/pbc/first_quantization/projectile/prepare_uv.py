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
r"""PREPARE the potential energy terms of the first quantized chemistry Hamiltonian with projectile.
"""
from functools import cached_property
from typing import Dict, TYPE_CHECKING

from attrs import frozen

from qualtran import Bloq, bloq_example, BloqBuilder, QAny, QBit, Register, Signature, SoquetT
from qualtran.bloqs.chemistry.pbc.first_quantization.prepare_zeta import PrepareZetaState
from qualtran.bloqs.chemistry.pbc.first_quantization.projectile.prepare_nu import (
    PrepareNuStateWithProj,
)

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@frozen
class PrepareUVFirstQuantizationWithProj(Bloq):
    r"""PREPARE the U and V parts of the Hamiltonian.

    Args:
        num_bits_p: The number of bits to represent each dimension of the momentum register.
        num_bits_n: The number of bits to represent each dimension of the
            momentum register for the projectile.
        eta: The number of electrons.
        num_atoms: The number of atoms. $L$ in the reference.
        m_param: $\mathcal{M}$ in the reference.
        lambda_zeta: sum of nuclear charges.
        num_bits_nuc_pos: The number of bits of precision for representing the nuclear coordinates.

    Registers:
        mu: The state controlling the nested boxes procedure.
        nu: The momentum transfer register.
        m: an ancilla register in a uniform superposition.
        l: The register for atomic species.
        flag_nu: Flag for success of the state preparation.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](https://arxiv.org/abs/2105.12767)
        page 19, section B
    """
    num_bits_p: int
    num_bits_n: int
    eta: int
    num_atoms: int
    m_param: int
    lambda_zeta: int
    num_bits_nuc_pos: int

    @cached_property
    def signature(self) -> Signature:
        n_m = (self.m_param - 1).bit_length()
        n_atom = (self.num_atoms - 1).bit_length()
        return Signature(
            [
                Register("mu", QAny(bitsize=self.num_bits_n)),
                Register("nu", QAny(bitsize=self.num_bits_n + 1), shape=(3,)),
                Register("m", QAny(bitsize=n_m)),
                Register("l", QAny(bitsize=n_atom)),
                Register("flag_nu", QBit()),
            ]
        )

    def pretty_name(self) -> str:
        return r'PREP UV'

    def build_composite_bloq(
        self, bb: BloqBuilder, mu: SoquetT, nu: SoquetT, m: SoquetT, l: SoquetT, flag_nu: SoquetT
    ) -> Dict[str, 'SoquetT']:
        mu, nu, m, flag_nu = bb.add(
            PrepareNuStateWithProj(self.num_bits_p, self.num_bits_n, self.m_param),
            mu=mu,
            nu=nu,
            m=m,
            flag_nu=flag_nu,
        )
        l = bb.add(PrepareZetaState(self.num_atoms, self.lambda_zeta, self.num_bits_nuc_pos), l=l)
        return {'mu': mu, 'nu': nu, 'm': m, 'l': l, 'flag_nu': flag_nu}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        # 1. Prepare the nu state
        # 2. Prepare the zeta_l state
        return {
            PrepareNuStateWithProj(self.num_bits_p, self.num_bits_n, self.m_param): 1,
            PrepareZetaState(self.num_atoms, self.lambda_zeta, self.num_bits_nuc_pos): 1,
        }


@bloq_example
def _prep_uv_proj() -> PrepareUVFirstQuantizationWithProj:
    num_bits_p = 6
    num_bits_n = 9
    eta = 10
    num_atoms = 10
    lambda_zeta = 10
    lambda_zeta = 10
    num_bits_nuc_pos = 8
    m_param = 2**19
    prep_uv_proj = PrepareUVFirstQuantizationWithProj(
        num_bits_p, num_bits_n, eta, num_atoms, m_param, lambda_zeta, num_bits_nuc_pos
    )
    return prep_uv_proj
