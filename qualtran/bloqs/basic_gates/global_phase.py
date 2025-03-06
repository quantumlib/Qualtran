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
from attrs import frozen

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
from qualtran.cirq_interop import CirqGateAsBloqBase
from qualtran.symbolics import pi, sarg, sexp, SymbolicComplex, SymbolicFloat

if TYPE_CHECKING:
    import quimb.tensor as qtn
    from pennylane.operation import Operation
    from pennylane.wires import Wires


@frozen(kw_only=True)
class GlobalPhase(CirqGateAsBloqBase):
    r"""Applies a global phase to the circuit as a whole.

    For an exponent $t$, the unitary effect is to multiply the state vector by the complex scalar
    $$
    (-1)^t = e^{i \pi t}
    $$

    The global phase of a state or circuit does not affect any observable quantity, but
    keeping track of it can be a useful bookkeeping mechanism for testing circuit identities, and
    the global phase becomes important when controlling an operation.

    This is fundamentally an atomic operation and this bloq has no decomposition in Qualtran.

    The single-qubit controlled version of `GlobalPhase` is `ZPowGate`.

    Args:
        exponent: the exponent t of the global phase (-1)^t to apply.
        eps: The precision of the rotation. This parameter is for bookkeeping and does
            not affect e.g. the tensor representation of this gate.
    """

    exponent: SymbolicFloat
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

    def as_pl_op(self, wires: 'Wires') -> 'Operation':
        import numpy as np
        import pennylane as qml

        return qml.GlobalPhase(phi=self.exponent * np.pi, wires=wires)

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
        if ctrl_spec != CtrlSpec():
            return super().get_ctrl_system(ctrl_spec=ctrl_spec)

        from qualtran.bloqs.basic_gates import ZPowGate

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


@bloq_example
def _global_phase() -> GlobalPhase:
    global_phase = GlobalPhase(exponent=0.5)
    return global_phase


_GLOBAL_PHASE_DOC = BloqDocSpec(
    bloq_cls=GlobalPhase, examples=[_global_phase], call_graph_example=None
)
