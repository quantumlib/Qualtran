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
r"""PREPARE the potential energy terms of the first quantized chemistry Hamiltonian."""
from functools import cached_property
from typing import Dict, Optional, Tuple, TYPE_CHECKING

from attrs import frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    QAny,
    QBit,
    Register,
    Signature,
    SoquetT,
)
from qualtran.bloqs.chemistry.pbc.first_quantization.prepare_nu import PrepareNuState
from qualtran.bloqs.chemistry.pbc.first_quantization.prepare_zeta import PrepareZetaState
from qualtran.drawing import Text, WireSymbol

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@frozen
class PrepareUVFirstQuantization(Bloq):
    r"""PREPARE the U and V parts of the Hamiltonian.

    Args:
        num_bits_p: The number of bits to represent each dimension of the momentum register.
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
    eta: int
    num_atoms: int
    m_param: int
    lambda_zeta: int
    num_bits_nuc_pos: int

    @cached_property
    def signature(self) -> Signature:
        # this is for the nu register which lives on a grid of twice the size
        # the nu grid is twice as large, so one more bit is needed
        n_m = (self.m_param - 1).bit_length()
        n_atom = (self.num_atoms - 1).bit_length()
        return Signature(
            [
                Register("mu", QAny(bitsize=self.num_bits_p)),
                Register("nu", QAny(bitsize=self.num_bits_p + 1), shape=(3,)),
                Register("m", QAny(bitsize=n_m)),
                Register("l", QAny(bitsize=n_atom)),
                Register("flag_nu", QBit()),
            ]
        )

    def build_composite_bloq(
        self, bb: BloqBuilder, mu: SoquetT, nu: SoquetT, m: SoquetT, l: SoquetT, flag_nu: SoquetT
    ) -> Dict[str, 'SoquetT']:
        mu, nu, m, flag_nu = bb.add(
            PrepareNuState(self.num_bits_p, self.m_param), mu=mu, nu=nu, m=m, flag_nu=flag_nu
        )
        l = bb.add(PrepareZetaState(self.num_atoms, self.lambda_zeta, self.num_bits_nuc_pos), l=l)
        return {'mu': mu, 'nu': nu, 'm': m, 'l': l, 'flag_nu': flag_nu}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        # 1. Prepare the nu state
        # 2. Prepare the zeta_l state
        return {
            PrepareNuState(self.num_bits_p, self.m_param): 1,
            PrepareZetaState(self.num_atoms, self.lambda_zeta, self.num_bits_nuc_pos): 1,
        }

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text('PREP UV')
        return super().wire_symbol(reg, idx)


@bloq_example
def _prepare_uv() -> PrepareUVFirstQuantization:
    num_bits_p = 5
    eta = 10
    num_atoms = 10
    lambda_zeta = 10
    m_param = 2**8
    num_bits_nuc_pos = 16

    prepare_uv = PrepareUVFirstQuantization(
        num_bits_p=num_bits_p,
        eta=eta,
        num_atoms=num_atoms,
        m_param=m_param,
        lambda_zeta=lambda_zeta,
        num_bits_nuc_pos=num_bits_nuc_pos,
    )
    return prepare_uv


_PREPARE_UV = BloqDocSpec(
    bloq_cls=PrepareUVFirstQuantization,
    import_line='from qualtran.bloqs.chemistry.pbc.first_quantization.prepare_uv import PrepareUVFirstQuantization',
    examples=(_prepare_uv,),
)
