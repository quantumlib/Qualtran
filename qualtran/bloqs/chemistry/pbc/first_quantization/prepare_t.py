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
r"""Bloqs for PREPARE T for the first quantized chemistry Hamiltonian."""
from functools import cached_property
from typing import Dict, Set, TYPE_CHECKING

from attrs import frozen

from qualtran import Bloq, bloq_example, BloqBuilder, BloqDocSpec, Signature, SoquetT
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.bloqs.prepare_uniform_superposition import PrepareUniformSuperposition

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@frozen
class PreparePowerTwoState(Bloq):
    r"""Prepares the uniform superposition over $|r\rangle$ given by Eq. 69 in the reference.

    This prepares the state

    $$
        2^{(-n_p -1)/2} \sum_r=0^{n_p-2} 2^{r/2} |r\rangle
    $$

    in one-hot unary.

    Args:
        bitsize: the number of bits $n_p$ for the $|r\rangle$ register.

    Registers:
        r: The register we want to prepare the state over.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization]
        (https://arxiv.org/abs/2105.12767) Eq 67-69, pg 19-20
    """
    bitsize: int

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(r=self.bitsize)

    def short_name(self) -> str:
        return r'PREP $2^{r/2} |r\rangle$'

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {(Toffoli(), (self.bitsize - 2))}


@frozen
class PrepareTFirstQuantization(Bloq):
    r"""PREPARE for the kinetic energy operator for the first quantized chemistry Hamiltonian.

    This prepares the state

    $$
        |+\rangle\sum_{j=1}^{\eta}|j\rangle\sum_{w=0}^{2}|w\rangle
        \sum_{r=0}^{n_{p}-2}2^{r/2}|r\rangle
        \sum_{s=0}^{n_{p}-2}2^{s/2}|s\rangle
    $$

    We assume that the uniform superposition over ($i$ and) $j$ has already occured via
    UniformSuperPositionIJFirstQuantization.

    Args:
        num_bits_p: The number of bits to represent each dimension of the momentum register.
        eta: The number of electrons.
        num_bits_rot_aa: The number of bits of precision for the single qubit
            rotation for amplitude amplification. Called $b_r$ in the reference.
        adjoint: whether to dagger the bloq or not.

    Registers:
        w: a register to index one of three components of the momenta.
        r: a register encoding bits for each component of the momenta.
        s: a register encoding bits for each component of the momenta.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](
            https://arxiv.org/abs/2105.12767) page 19, section B
    """

    num_bits_p: int
    eta: int
    num_bits_rot_aa: int = 8
    adjoint: int = False

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(w=2, r=self.num_bits_p, s=self.num_bits_p)

    def short_name(self) -> str:
        return r'PREP $T$'

    def build_composite_bloq(
        self, bb: BloqBuilder, w: SoquetT, r: SoquetT, s: SoquetT
    ) -> Dict[str, 'SoquetT']:
        w = bb.add(PrepareUniformSuperposition(3), target=w)
        r = bb.add(PreparePowerTwoState(self.num_bits_p), r=r)
        s = bb.add(PreparePowerTwoState(self.num_bits_p), r=s)
        return {'w': w, 'r': r, 's': s}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # there is a cost for the uniform state preparation for the $w$
        # register. Adding a bloq is sort of overkill, should just tag the
        # correct cost on UniformSuperPosition bloq
        # 13 is from assuming 8 bits for the rotation, and n = 2.
        uni_prep_w = (Toffoli(), 13)
        # Factor of two for r and s registers.
        return {uni_prep_w, (PreparePowerTwoState(bitsize=self.num_bits_p), 2)}


@bloq_example
def _prepare_t() -> PrepareTFirstQuantization:
    num_bits_p = 5
    eta = 10

    prepare_t = PrepareTFirstQuantization(num_bits_p=num_bits_p, eta=eta)
    return prepare_t


_PREPARE_T = BloqDocSpec(
    bloq_cls=PrepareTFirstQuantization,
    import_line='from qualtran.bloqs.chemistry.pbc.first_quantization.prepare_t import PrepareTFirstQuantization',
    examples=(_prepare_t,),
)
