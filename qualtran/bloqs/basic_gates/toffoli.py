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
from typing import Any, Dict, Set, Tuple, TYPE_CHECKING, Union

import numpy as np
from attrs import frozen

from qualtran import Bloq, bloq_example, BloqDocSpec, QBit, Register, Signature, Soquet, SoquetT
from qualtran.bloqs.basic_gates import TGate
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.resource_counting import SympySymbolAllocator

if TYPE_CHECKING:
    import cirq
    import quimb.tensor as qtn

    from qualtran.cirq_interop import CirqQuregT
    from qualtran.drawing import WireSymbol
    from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
    from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class Toffoli(Bloq):
    """The Toffoli gate.

    This will flip the target bit if both controls are active. It can be thought of as
    a reversible AND gate.

    Like `TGate`, this is a common compilation target. The Clifford+Toffoli gateset is
    universal.

    References:
        [Novel constructions for the fault-tolerant Toffoli gate](https://arxiv.org/abs/1212.5069).
        Cody Jones. 2012. Provides a decomposition into 4 `TGate`.
    """

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('ctrl', QBit(), shape=(2,)), Register('target', QBit())])

    def adjoint(self) -> 'Bloq':
        return self

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {(TGate(), 4)}

    def _t_complexity_(self):
        return TComplexity(t=4)

    def add_my_tensors(
        self,
        tn: 'qtn.TensorNetwork',
        tag: Any,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        import quimb.tensor as qtn

        from qualtran.bloqs.basic_gates.cnot import XOR

        # Set up the CTRL tensor which copies inputs to outputs and activates
        # when a==1 and b==1
        internal = qtn.rand_uuid()
        inds = (
            incoming['ctrl'][0],
            incoming['ctrl'][1],
            outgoing['ctrl'][0],
            outgoing['ctrl'][1],
            internal,
        )
        CTRL = np.zeros((2,) * 5, dtype=np.complex128)
        for a, b in itertools.product([0, 1], repeat=2):
            CTRL[a, b, a, b, int(a == 1 and b == 1)] = 1.0

        # Wire up the CTRL tensor to XOR to flip `target` when active.
        tn.add(qtn.Tensor(data=CTRL, inds=inds, tags=['COPY', tag]))
        tn.add(
            qtn.Tensor(
                data=XOR, inds=(incoming['target'], outgoing['target'], internal), tags=['XOR']
            )
        )

    def on_classical_vals(
        self, ctrl: 'ClassicalValT', target: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        assert target in [0, 1]
        if ctrl[0] == 1 and ctrl[1] == 1:
            target = (target + 1) % 2

        return {'ctrl': ctrl, 'target': target}

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', ctrl: 'CirqQuregT', target: 'CirqQuregT'  # type: ignore[type-var]
    ) -> Tuple[Union['cirq.Operation', None], Dict[str, 'CirqQuregT']]:  # type: ignore[type-var]
        import cirq

        (trg,) = target
        return cirq.CCNOT(*ctrl[:, 0], trg), {'ctrl': ctrl, 'target': target}

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        from qualtran.drawing import Circle, ModPlus

        if soq.reg.name == 'ctrl':
            return Circle(filled=True)
        elif soq.reg.name == 'target':
            return ModPlus()
        raise ValueError(f'Bad wire symbol soquet: {soq}')


@bloq_example
def _toffoli() -> Toffoli:
    toffoli = Toffoli()
    return toffoli


_TOFFOLI_DOC = BloqDocSpec(
    bloq_cls=Toffoli,
    import_line='from qualtran.bloqs.basic_gates import Toffoli',
    examples=[_toffoli],
)
