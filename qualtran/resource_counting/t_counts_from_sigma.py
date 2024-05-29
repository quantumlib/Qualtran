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
import inspect
import sys
from typing import Mapping, Optional, Tuple, Type, TYPE_CHECKING

import cirq

from qualtran.symbolics import ceil, SymbolicInt

if TYPE_CHECKING:
    from qualtran import Bloq
    from qualtran.bloqs.basic_gates.rotation import _HasEps


def _get_all_rotation_types() -> Tuple[Type['_HasEps'], ...]:
    """Returns all classes defined in bloqs.basic_gates which have an attribute `eps`."""
    from qualtran.bloqs.basic_gates import GlobalPhase
    from qualtran.bloqs.basic_gates.rotation import _HasEps

    bloqs_to_exclude = [GlobalPhase]

    return tuple(
        v
        for (_, v) in inspect.getmembers(sys.modules['qualtran.bloqs.basic_gates'], inspect.isclass)
        if isinstance(v, _HasEps) and v not in bloqs_to_exclude
    )


def t_counts_from_sigma(
    sigma: Mapping['Bloq', SymbolicInt],
    rotation_types: Optional[Tuple[Type['_HasEps'], ...]] = None,
) -> SymbolicInt:
    """Aggregates T-counts from a sigma dictionary by summing T-costs for all rotation bloqs."""
    from qualtran.bloqs.basic_gates import TGate
    from qualtran.cirq_interop.t_complexity_protocol import TComplexity

    if rotation_types is None:
        rotation_types = _get_all_rotation_types()
    ret = sigma.get(TGate(), 0) + sigma.get(TGate().adjoint(), 0)
    for bloq, counts in sigma.items():
        if isinstance(bloq, rotation_types) and not cirq.has_stabilizer_effect(bloq):
            ret += ceil(TComplexity.rotation_cost(bloq.eps)) * counts
    return ret
