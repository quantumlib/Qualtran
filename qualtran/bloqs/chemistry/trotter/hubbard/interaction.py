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
from typing import Set, TYPE_CHECKING, Union

import sympy
from attrs import frozen

from qualtran import Bloq, bloq_example, BloqDocSpec, QAny, Register, Signature
from qualtran.bloqs.basic_gates import Rz

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@frozen
class Interaction(Bloq):
    r"""Bloq implementing the hubbard U part of the hamiltonian.

    Specifically:
    $$
        U_I = e^{i t H_I}
    $$
    which can be implemented using equal angle single-qubit Z rotations.

    Args:
        length: Lattice length L.

    Registers:
        system: The system register of size 2 `length`.

    References:
        [Early fault-tolerant simulations of the Hubbard model](https://arxiv.org/abs/2012.09238)
        Eq. 6 page 2 and page 13 paragraph 1.
    """

    length: Union[int, sympy.Expr]
    angle: Union[float, sympy.Expr]
    eps: Union[float, sympy.Expr] = 1e-9

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('system', QAny(self.length), shape=(2,))])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # Page 13 paragraph 1.
        return {(Rz(angle=self.angle, eps=self.eps), self.length**2)}


@bloq_example
def _interaction() -> Interaction:
    length = 8
    angle = 0.5
    interaction = Interaction(length, angle)
    return interaction


_INTERACTION_DOC = BloqDocSpec(
    bloq_cls=Interaction,
    import_line='from qualtran.bloqs.chemistry.trotter.hubbard.interaction import Interaction',
    examples=(_interaction,),
)
