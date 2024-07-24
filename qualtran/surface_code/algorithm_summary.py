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

from typing import Optional, TYPE_CHECKING

from attrs import frozen

from qualtran.resource_counting import GateCounts, get_cost_value, QECGatesCost, QubitCount

if TYPE_CHECKING:
    from qualtran import Bloq


_QUBIT_COUNT = QubitCount()
_QEC_COUNT = QECGatesCost()


@frozen(kw_only=True)
class AlgorithmSummary:
    """Logical costs of a quantum algorithm that impact modeling of its physical cost.


    n_rotation_layers: In Qualtran, we don't actually push all the cliffords out and count
        the number of rotation layers, so this is just the number of rotations $M_R$ by default.
        If you are trying to reproduce numbers exactly, you can provide an explicit
        number of rotation layers.
    """

    n_algo_qubits: int
    n_logical_gates: GateCounts
    n_rotation_layers: Optional[int] = None

    @staticmethod
    def from_bloq(bloq: 'Bloq') -> 'AlgorithmSummary':
        gate_count = get_cost_value(bloq, _QEC_COUNT)
        qubit_count = int(get_cost_value(bloq, _QUBIT_COUNT))
        return AlgorithmSummary(n_algo_qubits=qubit_count, n_logical_gates=gate_count)
