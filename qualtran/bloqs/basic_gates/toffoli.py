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
from typing import Dict, Optional, Set, Tuple, TYPE_CHECKING, Union

from attrs import frozen

from qualtran import Bloq, Register, Signature
from qualtran.bloqs.basic_gates import TGate

if TYPE_CHECKING:
    import cirq

    from qualtran.cirq_interop import CirqQuregT
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
        return Signature([Register('ctrl', 1, shape=(2,)), Register('target', 1)])

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set['BloqCountT']:
        return {(4, TGate())}

    def on_classical_vals(
        self, ctrl: 'ClassicalValT', target: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:
        assert target in [0, 1]
        if ctrl[0] == 1 and ctrl[1] == 1:
            target = (target + 1) % 2

        return {'ctrl': ctrl, 'target': target}

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', ctrl: 'CirqQuregT', target: 'CirqQuregT'
    ) -> Tuple[Union['cirq.Operation', None], Dict[str, 'CirqQuregT']]:
        import cirq

        (trg,) = target
        return cirq.CCNOT(*ctrl[:, 0], trg), {'ctrl': ctrl, 'target': target}
