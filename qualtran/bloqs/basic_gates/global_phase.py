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
from functools import cached_property
from typing import Dict, Iterable, List, Sequence, Tuple, TYPE_CHECKING

import attrs
import cirq
from attrs import field, frozen

from qualtran import (
    AddControlledT,
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    CompositeBloq,
    ConnectionT,
    CtrlSpec,
    DecomposeTypeError,
    SoquetT,
)
from qualtran.bloqs.basic_gates.rotation import ZPowGate
from qualtran.cirq_interop import CirqGateAsBloqBase
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.symbolics import pi, sarg, sexp, SymbolicComplex, SymbolicFloat

if TYPE_CHECKING:
    import quimb.tensor as qtn


@frozen
class GlobalPhase(CirqGateAsBloqBase):
    r"""Applies a global phase to the circuit as a whole.

    The unitary effect is to multiply the state vector by the complex scalar
    $e^{i pi t}$ for `exponent` $t$.

    The global phase of a state or circuit does not affect any observable quantity, but
    keeping track of it can be a useful bookkeeping mechanism for testing circuit identities.
    The global phase becomes important if the gate becomes controlled.

    Args:
        exponent: the exponent $t$ of the global phase $e^{i pi t}$ to apply.
        eps: precision
    """

    exponent: SymbolicFloat = field(kw_only=True)
    eps: SymbolicFloat = 1e-11

    @cached_property
    def coefficient(self) -> SymbolicComplex:
        return sexp(self.exponent * pi(self.exponent) * 1j)

    @classmethod
    def from_coefficient(
        cls, coefficient: SymbolicComplex, *, eps: SymbolicFloat = 1e-11
    ) -> 'GlobalPhase':
        """Applies a global phase of `coefficient`."""
        return cls(exponent=sarg(coefficient) / pi(coefficient), eps=eps)

    @property
    def cirq_gate(self) -> cirq.Gate:
        return cirq.GlobalPhaseGate(self.coefficient)

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic")

    def adjoint(self) -> 'GlobalPhase':
        return attrs.evolve(self, exponent=-self.exponent)

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        import quimb.tensor as qtn

        return [qtn.Tensor(data=self.coefficient, inds=[], tags=[str(self)])]

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> Tuple['Bloq', 'AddControlledT']:
        # Delegate to superclass logic for more than one control.
        if not (ctrl_spec == CtrlSpec() or ctrl_spec == CtrlSpec(cvs=0)):
            return super().get_ctrl_system(ctrl_spec=ctrl_spec)

        # Otherwise, it's a ZPowGate
        if ctrl_spec == CtrlSpec(cvs=0):
            bloq = ZPowGate(exponent=-self.exponent, global_shift=-1, eps=self.eps)
        else:
            bloq = ZPowGate(exponent=self.exponent, eps=self.eps)

        def _add_ctrled(
            bb: 'BloqBuilder', ctrl_soqs: Sequence['SoquetT'], in_soqs: Dict[str, 'SoquetT']
        ) -> Tuple[Iterable['SoquetT'], Iterable['SoquetT']]:
            (ctrl,) = ctrl_soqs
            ctrl = bb.add(bloq, q=ctrl)
            return [ctrl], []

        return bloq, _add_ctrled

    def __str__(self) -> str:
        return f'GPhase({self.coefficient})'

    def _t_complexity_(self) -> 'TComplexity':
        return TComplexity()


@bloq_example
def _global_phase() -> GlobalPhase:
    global_phase = GlobalPhase(exponent=0.5)
    return global_phase


_GLOBAL_PHASE_DOC = BloqDocSpec(bloq_cls=GlobalPhase, examples=[_global_phase])
