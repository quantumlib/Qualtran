#  Copyright 2025 Google LLC
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
from typing import Dict, List, TYPE_CHECKING

from attrs import frozen

from qualtran import Bloq, CBit, ConnectionT, QBit, Register, Side, Signature
from qualtran.simulation.classical_sim import ClassicalValT

if TYPE_CHECKING:
    from qualtran.simulation.tensor import DiscardInd


@frozen
class Discard(Bloq):
    """Discard a classical bit.

    This is an allowed operation.
    """

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('c', CBit(), side=Side.LEFT)])

    def on_classical_vals(self, c: int) -> Dict[str, 'ClassicalValT']:
        return {}

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['DiscardInd']:

        from qualtran.simulation.tensor import DiscardInd

        return [DiscardInd((incoming['c'], 0))]


@frozen
class DiscardQ(Bloq):
    """Discard a qubit.

    This is a dangerous operation that can ruin your computation. This is equivalent to
    measuring the qubit and throwing out the measurement operation, so it removes any coherences
    involved with the qubit. Use with care.
    """

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('q', QBit(), side=Side.LEFT)])

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['DiscardInd']:

        from qualtran.simulation.tensor import DiscardInd

        return [DiscardInd((incoming['q'], 0))]
