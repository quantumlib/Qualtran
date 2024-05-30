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

from typing import TYPE_CHECKING

import attrs
import cirq
from attrs import frozen

from qualtran import bloq_example, BloqDocSpec, DecomposeTypeError
from qualtran.cirq_interop import CirqGateAsBloqBase
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.symbolics import sconj, SymbolicComplex

if TYPE_CHECKING:
    from qualtran import CompositeBloq


@frozen
class GlobalPhase(CirqGateAsBloqBase):
    """Global phase operation of $z$ (where $|z| = 1$).

    Args:
        coefficient: a unit complex number $z$ representing the global phase
        eps: precision
    """

    coefficient: 'SymbolicComplex'
    eps: float = 1e-11

    @property
    def cirq_gate(self) -> cirq.Gate:
        return cirq.GlobalPhaseGate(self.coefficient, self.eps)

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic")

    def adjoint(self) -> 'GlobalPhase':
        return attrs.evolve(self, coefficient=sconj(self.coefficient))

    def pretty_name(self) -> str:
        return 'GPhase'

    def __str__(self) -> str:
        return f'GPhase({self.coefficient})'

    def _t_complexity_(self) -> 'TComplexity':
        return TComplexity()


@bloq_example
def _global_phase() -> GlobalPhase:
    global_phase = GlobalPhase(1j)
    return global_phase


_GLOBAL_PHASE_DOC = BloqDocSpec(
    bloq_cls=GlobalPhase,
    import_line='from qualtran.bloqs.basic_gates import GlobalPhase',
    examples=[_global_phase],
)
