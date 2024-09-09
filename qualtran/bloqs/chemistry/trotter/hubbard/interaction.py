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
r"""Bloqs implementing unitary evolution under the interacting part of the Hubbard Hamiltonian."""

from functools import cached_property
from typing import TYPE_CHECKING

from attrs import frozen

from qualtran import Bloq, bloq_example, BloqDocSpec, QAny, Register, Signature
from qualtran.bloqs.basic_gates import Rz
from qualtran.bloqs.rotations.hamming_weight_phasing import HammingWeightPhasing
from qualtran.symbolics import SymbolicFloat, SymbolicInt

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


@frozen
class Interaction(Bloq):
    r"""Bloq implementing the hubbard U part of the hamiltonian.

    Specifically:
    $$
        U_I = e^{i t H_I}
    $$
    which can be implemented using equal angle single-qubit Z rotations.

    Args:
        length: Lattice length $L$.
        angle: The prefactor scaling the Hopping hamiltonian in the unitary (`t` above).
            This should contain any relevant prefactors including the time step
            and any splitting coefficients.
        hubb_u: The hubbard $U$ parameter.
        eps: The precision of the single qubit rotations.

    Registers:
        system: The system register of size 2 `length`.

    References:
        [Early fault-tolerant simulations of the Hubbard model](https://arxiv.org/abs/2012.09238)
        Eq. 6 page 2 and page 13 paragraph 1.
    """

    length: SymbolicInt
    angle: SymbolicFloat
    hubb_u: SymbolicFloat
    eps: SymbolicFloat = 1e-9

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('system', QAny(self.length), shape=(2,))])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        # Page 13 paragraph 1.
        return {Rz(angle=self.angle * self.hubb_u, eps=self.eps): self.length**2}


@frozen
class InteractionHWP(Bloq):
    r"""Bloq implementing the hubbard U part of the hamiltonian using Hamming weight phasing.

    Specifically:
    $$
        U_I = e^{i t H_I}
    $$
    which can be implemented using equal angle single-qubit Z rotations.

    Each interaction term can be implemented using a e^{iZZ} gate, which
    decomposes into a single Rz gate flanked by cliffords. There are L^2
    equal angle rotations in total all of which may be applied in parallel using HWP.

    Args:
        length: Lattice length L.
        angle: The rotation angle for unitary.
        hubb_u: The hubbard U parameter.
        eps: The precision for single qubit rotations.

    Registers:
        system: The system register of size 2 `length`.

    References:
        [Early fault-tolerant simulations of the Hubbard model](
            https://arxiv.org/abs/2012.09238) Eq. page 13 paragraph 1, and page
            14 paragraph 3 right column. The apply 2 batches of $L^2/2$ rotations.
    """

    length: SymbolicInt
    angle: SymbolicFloat
    hubb_u: SymbolicFloat
    eps: SymbolicFloat = 1e-9

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('system', QAny(self.length), shape=(2,))])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return {
            HammingWeightPhasing(self.length**2 // 2, self.angle * self.hubb_u, eps=self.eps): 2
        }


@bloq_example
def _interaction() -> Interaction:
    length = 8
    angle = 0.5
    hubb_u = 4.0
    interaction = Interaction(length, angle, hubb_u)
    return interaction


_INTERACTION_DOC = BloqDocSpec(
    bloq_cls=Interaction,
    import_line='from qualtran.bloqs.chemistry.trotter.hubbard.interaction import Interaction',
    examples=(_interaction,),
)


@bloq_example
def _interaction_hwp() -> InteractionHWP:
    length = 8
    angle = 0.5
    hubb_u = 4.0
    interaction_hwp = InteractionHWP(length, angle, hubb_u)
    return interaction_hwp


_INTERACTION_HWP_DOC = BloqDocSpec(
    bloq_cls=InteractionHWP,
    import_line='from qualtran.bloqs.chemistry.trotter.hubbard.interaction import InteractionHWP',
    examples=(_interaction_hwp,),
)
