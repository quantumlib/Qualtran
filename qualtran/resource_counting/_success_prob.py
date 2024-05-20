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
import abc
import logging
from typing import Callable, Dict, Generic, Sequence, Tuple, TYPE_CHECKING

import sympy
from attrs import frozen

from qualtran import Bloq, DecomposeNotImplementedError, DecomposeTypeError

from . import CostValT
from ._call_graph import get_bloq_callee_counts
from ._costing import CostKey

logger = logging.getLogger(__name__)


@frozen
class SuccessProb(CostKey[float]):
    def compute(self, bloq: 'Bloq', get_callee_cost: Callable[['Bloq'], float]) -> float:
        tot: float = 1.0
        callees = get_bloq_callee_counts(bloq)
        logger.info("Computing %s for %s from %d callee(s)", self, bloq, len(callees))
        for callee, n in callees:
            v = get_callee_cost(callee)
            tot *= v**n
        return tot

    def zero(self) -> CostValT:
        return 1.0  # under multiplication, 1 is the identity.

    def __str__(self):
        return 'success prob'
