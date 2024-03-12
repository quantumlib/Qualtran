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
from typing import Dict, Optional, Tuple, Union

import sympy

from qualtran import Adjoint, Bloq
from qualtran.bloqs.basic_gates import CSwap, TGate, Toffoli
from qualtran.bloqs.reflection import Reflection
from qualtran.resource_counting.generalizers import (
    ignore_alloc_free,
    ignore_cliffords,
    ignore_partition,
    ignore_split_join,
)
from qualtran.resource_counting.t_counts_from_sigma import (
    _get_all_rotation_types,
    t_counts_from_sigma,
)


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
    return bloq_classifier


def classify_bloq(bloq: Bloq, bloq_classification: Dict[str, Tuple[Bloq]]) -> str:
    """Classify a bloq given a bloq_classification.

    Args:
        bloq: The bloq to classify
        bloq_classification: A dictionary mapping a classification to a tuple of
            bloqs in that classification.
    Returns:
        classification: The matching key in bloq_classification. Returns other if not classified.
    """
    for k, v in bloq_classification.items():
        if isinstance(bloq, v):
            return k
        elif isinstance(bloq, Adjoint) and isinstance(bloq.subbloq, v):
            return k
    return 'other'


def _keep_only_classified_bloqs(bloq: Bloq, bloq_classification: Dict[str, Tuple[Bloq]]) -> bool:
    """A keep method for a bloqs call graph to turn classified bloqs into leaf nodes."""
    for _, v in bloq_classification.items():
        if isinstance(bloq, v):
            return True
        elif isinstance(bloq, Adjoint) and isinstance(bloq.adjoint(), v):
            return True
    return False


def classify_t_count_by_bloq_type(
    bloq: Bloq, bloq_classification: Optional[Dict[str, Tuple[Bloq]]] = None
) -> Dict[str, Union[int, sympy.Expr]]:
    """Classify (bin) the T count of a bloq's call graph by type of operation.

    Args:
        bloq: the bloq to classify.
        bloq_classification: An optional dictionary mapping bloq_classifications to bloq types.

    Returns
        classified_bloqs: dictionary containing the T count for different types of bloqs.
    """
    if bloq_classification is None:
        bloq_classification = _get_basic_bloq_classification()
    keeper = lambda bloq: _keep_only_classified_bloqs(bloq, bloq_classification)
    _, sigma = bloq.call_graph(
        generalizer=[ignore_split_join, ignore_alloc_free, ignore_cliffords, ignore_partition],
        keep=keeper,
    )
    classified_bloqs = defaultdict(int)
    for k, v in sigma.items():
        classification = classify_bloq(k, bloq_classification)
        classified_bloqs[classification] += v * t_counts_from_sigma(k.call_graph()[1])
    return classified_bloqs
