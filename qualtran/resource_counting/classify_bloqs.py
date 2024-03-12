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
import importlib
import inspect
import sys
from collections import defaultdict
from typing import Callable, Dict, Optional, Sequence, Tuple, TYPE_CHECKING, Union

from qualtran import Bloq
from qualtran.bloqs.basic_gates import CSwap, TGate, Toffoli
from qualtran.bloqs.reflection import Reflection
from qualtran.bloqs.util_bloqs import Allocate, Free, Join, Split
from qualtran.resource_counting.t_counts_from_sigma import _get_all_rotation_types

if TYPE_CHECKING:
    from qualtran import Bloq
    from qualtran.resource_counting import BloqCountT, GeneralizerT, SympySymbolAllocator


def _get_all_bloqs_in_module(bloq_module: str) -> Tuple[Bloq]:
    """Returns all classes defined in bloqs `bloq_module`."""
    importlib.import_module(bloq_module)
    all_members = [v for (_, v) in inspect.getmembers(sys.modules[bloq_module], inspect.isclass)]
    defined_in_module = tuple(member for member in all_members if bloq_module in member.__module__)
    return defined_in_module


def _get_basic_bloq_classification() -> Dict[str, Tuple[Bloq]]:
    """High level classification of bloqs by the module name."""

    bloq_classifier = {
        'arithmetic': _get_all_bloqs_in_module('qualtran.bloqs.arithmetic'),
        'rotations': _get_all_bloqs_in_module('qualtran.bloqs.rotations')
        + _get_all_rotation_types(),
        'state_preparation': _get_all_bloqs_in_module('qualtran.bloqs.state_preparation'),
        'data_loading': _get_all_bloqs_in_module('qualtran.bloqs.data_loading'),
        'mcmt': _get_all_bloqs_in_module('qualtran.bloqs.mcmt'),
        'multiplexers': _get_all_bloqs_in_module('qualtran.bloqs.multiplexers'),
        'swaps': _get_all_bloqs_in_module('qualtran.bloqs.swap_network') + (CSwap,),
        'reflection': (Reflection,),
        'toffoli': (Toffoli,),
        'tgate': (TGate,),
    }


def keeper(bloq: Bloq, classification: Dict[str, Tuple[Bloq]]) -> bool:
    for k, v in _get_basic_bloq_classification().items():
        if isinstance(bloq, v):
            return True
    return False


def classify_t_count_by_bloq_type(
    bloq: Bloq,
    generalizer: Optional[Union['GeneralizerT', Sequence['GeneralizerT']]] = None,
    bloq_classification: Optional[Dict[str, Tuple[Bloq]]] = None,
) -> Dict[str, int]:
    """Classify (bin) the T count of a bloq's call graph by type of operation.

    Args:
        bloq: the bloq to classify.
        generalizer: If provided, run this function on each (sub)bloq to replace attributes
            that do not affect resource estimates with generic sympy symbols. If the function
            returns `None`, the bloq is omitted from the counts graph. If a sequence of
            generalizers is provided, each generalizer will be run in order.
        bloq_classification: An optional dictionary mapping bloq_classifications to bloq types.

    Returns
        classified_bloqs: dictionary containing the T count for different types of bloqs.
    """
    classified_bloqs = defaultdict(int)
    if bloq_classification is None:
        bloq_classification = _get_basic_bloq_classification()
    for bloq, num_calls in bloq.bloq_counts().items():
        # only classify bloqs which contribute to the T count.
        if isinstance(bloq, (Split, Join, Allocate, Free)):
            continue
        num_t = bloq.call_graph(generalizer=generalizer)[1].get(TGate())
        if num_t is not None:
            num_t = int(num_t)
            is_classified = False
            for k, v in bloq_classification.items():
                if isinstance(bloq, v):
                    classified_bloqs[k] += num_calls * num_t
                    is_classified = True
            if not is_classified:
                classified_bloqs['other'] += num_calls * num_t
    return classified_bloqs
