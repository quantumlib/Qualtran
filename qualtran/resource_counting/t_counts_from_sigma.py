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
from typing import Dict, Optional, Tuple, TYPE_CHECKING, Union

import cirq

from qualtran.bloqs.basic_gates import TGate
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.resource_counting.symbolic_counting_utils import SymbolicInt

if TYPE_CHECKING:
    import sympy

    from qualtran import Bloq
    from qualtran.bloqs.basic_gates.rotation import _HasEps


def _get_all_rotation_types() -> Tuple['_HasEps', ...]:
    """Returns all classes defined in bloqs.basic_gates which have an attribute `eps`."""
    import qualtran.bloqs.basic_gates  # pylint: disable=unused-import

    return tuple(
        v
        for (k, v) in inspect.getmembers(sys.modules['qualtran.bloqs.basic_gates'], inspect.isclass)
        if hasattr(v, 'eps')
    )


def t_counts_from_sigma(
    sigma: Dict['Bloq', Union[int, 'sympy.Expr']],
    rotation_types: Optional[Tuple['_HasEps', ...]] = None,
) -> SymbolicInt:
    """Aggregates T-counts from a sigma dictionary by summing T-costs for all rotation bloqs."""
    if rotation_types is None:
        rotation_types = _get_all_rotation_types()
    ret = sigma.get(TGate(), 0)
    for bloq, counts in sigma.items():
        if isinstance(bloq, rotation_types) and not cirq.has_stabilizer_effect(bloq):
            ret += TComplexity.rotation_cost(bloq.eps) * counts
    return ret
