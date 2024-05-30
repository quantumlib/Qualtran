#  Copyright 2024 Google LLC
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

import logging
from typing import Callable, Dict, TYPE_CHECKING

from attrs import frozen

from ._bloq_counts import bloq_is_clifford
from ._call_graph import get_bloq_callee_counts
from ._costing import CostKey

if TYPE_CHECKING:
    from qualtran import Bloq

AlgorithmSummaryDict = Dict[str, int]

logger = logging.getLogger(__name__)


@frozen
class AlgorithmSummaryCounts(CostKey[AlgorithmSummaryDict]):
    """A cost that represents the variables that affect physical resource estimation.

    The result of computing this cost is a mapping from arguments of AlgorithmSummary to values
    with two exceptions:
        - `algorithm_qubits` are not counted and should be computed using QubitCount cost.
        - `measurements` are not counted since qualtran doesn't measurement as a bloq.
    """

    def compute(
        self, bloq: 'Bloq', get_callee_cost: Callable[['Bloq'], AlgorithmSummaryDict]
    ) -> AlgorithmSummaryDict:
        from qualtran.bloqs.basic_gates import TGate, Toffoli, TwoBitCSwap
        from qualtran.bloqs.mcmt.and_bloq import And

        # Cliffords
        if bloq_is_clifford(bloq):
            return self.zero()

        # T gates
        if isinstance(bloq, TGate):
            return {'t_gates': 1}

        # Toffolis
        if isinstance(bloq, Toffoli):
            return {'toffoli_gates': 1}

        # 'And' bloqs
        if isinstance(bloq, And):
            return {'toffoli_gates': int(not bloq.uncompute)}

        # CSwaps aka Fredkin
        if isinstance(bloq, TwoBitCSwap):
            return {'toffoli_gates': 1}

        callees = get_bloq_callee_counts(bloq)
        if len(callees) == 0:
            # If not Clifford or T of Toffoli and has no callees: assume a rotation.
            return {'rotation_gates': 1}

        # Recursive case
        logger.info("Computing %s for %s from %d callee(s)", self, bloq, len(callees))
        totals = {}
        depth = 0
        for callee, n_times_called in callees:
            callee_cost = get_callee_cost(callee)
            for k, v in callee_cost.items():
                totals[k] = totals.get(k, 0) + n_times_called * v
            depth = max(depth, callee_cost.get('rotation_circuit_depth', 0) + 1)
        totals['rotation_circuit_depth'] = depth
        return totals

    def zero(self) -> AlgorithmSummaryDict:
        return {}
