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
from collections import defaultdict
from typing import Dict, Optional, Tuple, Union

import sympy

from qualtran import Bloq
from qualtran.resource_counting.generalizers import (
    ignore_alloc_free,
    ignore_cliffords,
    ignore_partition,
    ignore_split_join,
)
from qualtran.resource_counting.t_counts_from_sigma import t_counts_from_sigma


def _get_basic_bloq_classification() -> Dict[str, Tuple[str]]:
    """High level classification of bloqs by the module name."""
    bloq_classifier = {
        'arithmetic': ('qualtran.bloqs.arithmetic',),
        'rotations': ('qualtran.bloqs.rotations',),
        'state_preparation': ('qualtran.bloqs.state_preparation',),
        'data_loading': ('qualtran.bloqs.data_loading',),
        'mcmt': ('qualtran.bloqs.mcmt',),
        'multiplexers': ('qualtran.bloqs.multiplexers',),
        'swaps': ('qualtran.bloqs.swap_network', 'qualtran.bloqs.basic_gates.swap'),
        'reflection': ('qualtran.bloqs.reflection',),
        'toffoli': ('qualtran.bloqs.basic_gates.toffoli',),
        'tgate': ('qualtran.bloqs.basic_gates.t_gate',),
    }
    return bloq_classifier


def classify_bloq(bloq: Bloq, bloq_classification: Dict[str, Tuple[str]]) -> str:
    """Classify a bloq given a bloq_classification.

    Args:
        bloq: The bloq to classify
        bloq_classification: A dictionary mapping a classification to a tuple of
            bloqs in that classification.
    Returns:
        classification: The matching key in bloq_classification. Returns other if not classified.
    """
    for k, v in bloq_classification.items():
        for mod in v:
            if mod in bloq.__module__:
                return k
            elif 'adjoint' in bloq.__module__ and mod in bloq.subbloq.__module__:
                return k
    return 'other'


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
    keeper = lambda bloq: classify_bloq(bloq, bloq_classification) != 'other'
    _, sigma = bloq.call_graph(
        generalizer=[ignore_split_join, ignore_alloc_free, ignore_cliffords, ignore_partition],
        keep=keeper,
    )
    classified_bloqs = defaultdict(int)
    for k, v in sigma.items():
        classification = classify_bloq(k, bloq_classification)
        classified_bloqs[classification] += v * t_counts_from_sigma(k.call_graph()[1])
    return classified_bloqs
