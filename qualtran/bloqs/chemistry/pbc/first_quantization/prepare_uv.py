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
from typing import Dict, Set, TYPE_CHECKING

from attrs import frozen

from qualtran import Bloq, BloqBuilder, Register, Signature, SoquetT
from qualtran.bloqs.chemistry.pbc.first_quantization.prepare_nu import PrepareNuState
from qualtran.bloqs.chemistry.pbc.first_quantization.prepare_zeta import PrepareZetaState

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@frozen
class PrepareUVFistQuantization(Bloq):
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
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](
            https://arxiv.org/abs/2105.12767) page 19, section B
    """
    num_bits_p: int
    eta: int
    num_atoms: int
    m_param: int
    lambda_zeta: int
    num_bits_nuc_pos: int
    adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        # this is for the nu register which lives on a grid of twice the size
        # the nu grid is twice as large, so one more bit is needed
        n_m = (self.m_param - 1).bit_length()
        n_atom = (self.num_atoms - 1).bit_length()
        return Signature(
            [
                Register("mu", bitsize=self.num_bits_p),
                Register("nu", bitsize=self.num_bits_p + 1, shape=(3,)),
                Register("m", bitsize=n_m),
                Register("l", bitsize=n_atom),
                Register("flag_nu", bitsize=1),
            ]
        )

    def build_composite_bloq(
        self, bb: BloqBuilder, mu: SoquetT, nu: SoquetT, m: SoquetT, l: SoquetT, flag_nu: SoquetT
    ) -> Dict[str, 'SoquetT']:
        mu, nu, m, flag_nu = bb.add(
            PrepareNuState(self.num_bits_p, self.m_param, adjoint=self.adjoint),
            mu=mu,
            nu=nu,
            m=m,
            flag_nu=flag_nu,
        )
        l = bb.add(
            PrepareZetaState(
                self.num_atoms, self.lambda_zeta, self.num_bits_nuc_pos, adjoint=self.adjoint
            ),
            l=l,
        )
        return {'mu': mu, 'nu': nu, 'm': m, 'l': l, 'flag_nu': flag_nu}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # 1. Prepare the nu state
        # 2. Prepare the zeta_l state
        return {
            (PrepareNuState(self.num_bits_p, self.m_param, self.adjoint), 1),
            (
                PrepareZetaState(
                    self.num_atoms, self.lambda_zeta, self.num_bits_nuc_pos, self.adjoint
                ),
                1,
            ),
        }
