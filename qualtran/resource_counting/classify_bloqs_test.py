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
from functools import cached_property
from typing import Set, TYPE_CHECKING

import attrs
import pytest

from qualtran import Bloq, QInt, Signature
from qualtran.bloqs.arithmetic import Add
from qualtran.bloqs.arithmetic.comparison import LessThanConstant
from qualtran.bloqs.basic_gates import CSwap, TGate
from qualtran.bloqs.data_loading.qrom import QROM
from qualtran.bloqs.mcmt.and_bloq import And
from qualtran.bloqs.reflection import Reflection
from qualtran.bloqs.rotations.hamming_weight_phasing import HammingWeightPhasing
from qualtran.resource_counting import BloqCountT
from qualtran.resource_counting.classify_bloqs import (
    _get_all_bloqs_in_module,
    classify_t_count_by_bloq_type,
)

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountT, GeneralizerT, SympySymbolAllocator


@attrs.frozen
class TestBundleOfBloqs(Bloq):
    """A fake bloq which just defines a call graph"""

    bloqs: BloqCountT

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build()

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return self.bloqs


@pytest.mark.parametrize(
    'bloq_count, classification',
    (
        (((CSwap(10), 42),), 'swaps'),
        (((HammingWeightPhasing(10, 1.11), 11),), 'rotations'),
        (((Add(QInt(8)), 4),), 'arithmetic'),
        (((QROM.build([4, 10, 11, 34]), 8),), 'data_loading'),
        (((And(), 4),), 'mcmt'),
        (((Reflection((3, 3, 2), (0, 0, 1)), 100),), 'reflection'),
        (((LessThanConstant(8, 3), 10),), 'arithmetic'),
    ),
)
def test_default_classification(bloq_count, classification):
    bloq = TestBundleOfBloqs(bloq_count)
    classified_bloqs = classify_t_count_by_bloq_type(bloq)
    bc = bloq_count[0]
    assert classified_bloqs[classification] == bc[1] * bc[0].call_graph()[1].get(TGate())


def test_get_all_bloqs_in_module():
    comparitors = _get_all_bloqs_in_module('qualtran.bloqs.arithmetic.comparison')
    assert isinstance(LessThanConstant(8, 3), comparitors)
    arithmetic = _get_all_bloqs_in_module('qualtran.bloqs.arithmetic')
    assert isinstance(LessThanConstant(8, 3), arithmetic)
    rotation = _get_all_bloqs_in_module('qualtran.bloqs.rotations')
    assert not isinstance(LessThanConstant(8, 3), rotation)
    assert isinstance(HammingWeightPhasing(10, 1.11), rotation)
    data_load = _get_all_bloqs_in_module('qualtran.bloqs.data_loading')
    assert isinstance(QROM.build([4, 10, 11, 23]), data_load)


def test_classify_bloq_counts_with_custom_bloq_classification():
    bloq_classification = {'swaps': (CSwap,)}
    test_bloq = TestBundleOfBloqs(((CSwap(10), 42), (Add(QInt(4)), 3)))
    classified_bloqs = classify_t_count_by_bloq_type(
        test_bloq, bloq_classification=bloq_classification
    )
    assert classified_bloqs == {'swaps': 42 * 10 * 7, 'other': 3 * 4 * (4 - 1)}
    assert test_bloq.call_graph()[1].get(TGate()) == sum(classified_bloqs.values())
