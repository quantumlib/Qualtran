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
from typing import Mapping, TYPE_CHECKING

import cirq

from qualtran.symbolics import ceil, SymbolicInt

if TYPE_CHECKING:
    from qualtran import Bloq


def t_counts_from_sigma(sigma: Mapping['Bloq', SymbolicInt]) -> SymbolicInt:
    """Aggregates T-counts from a sigma dictionary by summing T-costs for all rotation bloqs."""
    from qualtran.bloqs.basic_gates import TGate
    from qualtran.cirq_interop.t_complexity_protocol import TComplexity
    from qualtran.resource_counting.classify_bloqs import bloq_is_rotation

    ret = sigma.get(TGate(), 0) + sigma.get(TGate().adjoint(), 0)
    for bloq, counts in sigma.items():
        if bloq_is_rotation(bloq) and not cirq.has_stabilizer_effect(bloq):
            assert hasattr(bloq, 'eps')
            ret += ceil(TComplexity.rotation_cost(bloq.eps)) * counts
    return ret
