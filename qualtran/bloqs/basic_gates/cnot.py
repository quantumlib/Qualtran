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

import itertools
from functools import cached_property
from typing import Any, Dict, Tuple, TYPE_CHECKING

import numpy as np
import quimb.tensor as qtn
from attrs import frozen

from qualtran import (
    Bloq,
    bloq_example,
    CompositeBloq,
    DecomposeTypeError,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.drawing import Circle, ModPlus, WireSymbol

if TYPE_CHECKING:
    import cirq

    from qualtran.cirq_interop import CirqQuregT
    from qualtran.simulation.classical_sim import ClassicalValT


COPY = [1, 0, 0, 0, 0, 0, 0, 1]
COPY = np.array(COPY, dtype=np.complex128).reshape((2, 2, 2))

XOR = np.array(list(itertools.product([0, 1], repeat=3)))
XOR = 1 - np.sum(XOR, axis=1) % 2
XOR = XOR.reshape((2, 2, 2)).astype(np.complex128)


@frozen
class CNOT(Bloq):
    """Two-qubit controlled-NOT.

    Registers:
        ctrl: One-bit control register.
        target: One-bit target register.
    """

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(ctrl=1, target=1)

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic")

    def adjoint(self) -> 'Bloq':
        return self

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        tag: Any,
        *,
        incoming: Dict[str, SoquetT],
        outgoing: Dict[str, SoquetT],
    ):
        """Append tensors to `tn` that represent this operation.

        This bloq uses the factored form of CNOT composed of a COPY and XOR tensor joined
        by an internal index.

        References:
            [Lectures on Quantum Tensor Networks](https://arxiv.org/abs/1912.10049). Biamonte 2019.
        """
        internal = qtn.rand_uuid()
        tn.add(
            qtn.Tensor(
                data=COPY, inds=(incoming['ctrl'], outgoing['ctrl'], internal), tags=['COPY', tag]
            )
        )
        tn.add(
            qtn.Tensor(
                data=XOR, inds=(incoming['target'], outgoing['target'], internal), tags=['XOR']
            )
        )

    def on_classical_vals(self, ctrl: int, target: int) -> Dict[str, 'ClassicalValT']:
        return {'ctrl': ctrl, 'target': (ctrl + target) % 2}

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', ctrl: 'CirqQuregT', target: 'CirqQuregT'
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        import cirq

        (ctrl,) = ctrl
        (target,) = target
        return cirq.CNOT(ctrl, target), {'ctrl': np.array([ctrl]), 'target': np.array([target])}

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        if soq.reg.name == 'ctrl':
            return Circle(filled=True)
        elif soq.reg.name == 'target':
            return ModPlus()
        raise ValueError(f'Bad wire symbol soquet: {soq}')

    def t_complexity(self) -> 'TComplexity':
        return TComplexity(clifford=1)


@bloq_example
def _cnot() -> CNOT:
    cnot = CNOT()
    return cnot
