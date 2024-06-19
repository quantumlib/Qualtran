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

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import attrs
import cirq
import numpy as np
from attrs import frozen

from qualtran import (
    AddControlledT,
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    ConnectionT,
    CtrlSpec,
    DecomposeTypeError,
    SoquetT,
)
from qualtran.cirq_interop import CirqGateAsBloqBase
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.symbolics import sconj, SymbolicComplex

from .rotation import ZPowGate

if TYPE_CHECKING:
    import quimb.tensor as qtn

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

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        import quimb.tensor as qtn

        return [qtn.Tensor(data=self.coefficient, inds=[], tags=[str(self)])]

    def get_ctrl_system(
        self, ctrl_spec: Optional['CtrlSpec'] = None
    ) -> Tuple['Bloq', 'AddControlledT']:

        # Default logic for more than one control.
        if not (ctrl_spec is None or ctrl_spec == CtrlSpec()):
            return super().get_ctrl_system(ctrl_spec=ctrl_spec)

        exponent: float = np.real_if_close(np.log(self.coefficient) / (np.pi * 1.0j)).item()  # type: ignore
        bloq = ZPowGate(exponent=exponent, eps=self.eps)

        def _add_ctrled(
            bb: 'BloqBuilder', ctrl_soqs: Sequence['SoquetT'], in_soqs: Dict[str, 'SoquetT']
        ) -> Tuple[Iterable['SoquetT'], Iterable['SoquetT']]:
            (ctrl,) = ctrl_soqs
            ctrl = bb.add(bloq, q=ctrl)
            return [ctrl], []

        return bloq, _add_ctrled

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
