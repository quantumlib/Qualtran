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
r"""PREPARE the superposition over nuclear weights for the first quantized chemistry Hamiltonian.
"""
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from attrs import evolve, frozen

from qualtran import Bloq, QAny, Register, Signature
from qualtran.bloqs.basic_gates import Toffoli

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@frozen
class PrepareZetaState(Bloq):
    r"""PREPARE the superpostion over $l$ weighted by $\zeta_l$.

    See https://github.com/quantumlib/Qualtran/issues/473.
    Args:
        num_bits_p: The number of bits to represent each dimension of the momentum register.
        eta: The number of electrons.
        m_param: $\mathcal{M}$ in the reference.
        lambda_zeta: sum of nuclear charges.

    Registers:
        l: the register indexing the atomic number.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](https://arxiv.org/abs/2105.12767)
        page 23-24, last 3 paragraphs.
    """
    num_atoms: int
    lambda_zeta: int
    num_bits_nuc_pos: int
    is_adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register("l", QAny(bitsize=(self.num_atoms - 1).bit_length()))])

    def adjoint(self) -> 'Bloq':
        return evolve(self, is_adjoint=not self.is_adjoint)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        if self.is_adjoint:
            # Really Er(x), eq 91. In practice we will replace this with the
            # appropriate qrom call down the line.
            return {Toffoli(): int(np.ceil(self.lambda_zeta**0.5))}
        else:

            return {Toffoli(): self.lambda_zeta}
