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
from typing import Tuple

import attrs
import numpy as np
import pytest

from qualtran import Bloq, CtrlSpec, QInt, Signature
from qualtran.bloqs.arithmetic import Add
from qualtran.bloqs.arithmetic.comparison import LessThanConstant
from qualtran.bloqs.basic_gates import CSwap, Rx, Rz, TGate, YPowGate
from qualtran.bloqs.data_loading.qrom import QROM
from qualtran.bloqs.mcmt.and_bloq import And
from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity
from qualtran.bloqs.reflections.reflection_using_prepare import ReflectionUsingPrepare
from qualtran.bloqs.rotations.hamming_weight_phasing import HammingWeightPhasing
from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
    PrepareUniformSuperposition,
)
from qualtran.resource_counting import (
    BloqCountDictT,
    BloqCountT,
    get_cost_value,
    QECGatesCost,
    SympySymbolAllocator,
)
from qualtran.resource_counting.classify_bloqs import (
    _get_basic_bloq_classification,
    bloq_is_rotation,
    bloq_is_t_like,
    classify_bloq,
    classify_t_count_by_bloq_type,
)


@attrs.frozen
class TestBundleOfBloqs(Bloq):
    """A fake bloq which just defines a call graph"""

    bloqs: Tuple[BloqCountT, ...]

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build()

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        return dict(self.bloqs)


@pytest.mark.parametrize(
    'bloq_count, classification',
    (
        (((CSwap(10), 42),), 'swaps'),
        (((HammingWeightPhasing(10, 1.11), 11),), 'rotations'),
        (((Add(QInt(8)), 4),), 'arithmetic'),
        (((QROM.build_from_data([4, 10, 11, 34]), 8),), 'data_loading'),
        (((And(), 4),), 'multi_control_pauli'),
        # https://github.com/python/mypy/issues/5313
        (((ReflectionUsingPrepare(PrepareIdentity.from_bitsizes((3, 3, 2))), 100),), 'reflection'),  # type: ignore[arg-type]
        (((LessThanConstant(8, 3), 10),), 'arithmetic'),
    ),
)
def test_default_classification(bloq_count, classification):
    bloq = TestBundleOfBloqs(bloq_count)
    classified_bloqs = classify_t_count_by_bloq_type(bloq)
    assert classified_bloqs == {
        classification: get_cost_value(bloq, QECGatesCost()).total_t_count()
    }


def test_dont_return_zeros():
    # Requires zero Ts if n is a power of 2
    bloqs = TestBundleOfBloqs(((PrepareUniformSuperposition(4), 2),))
    classified_bloqs = classify_t_count_by_bloq_type(bloqs)
    assert len(classified_bloqs) == 0


@pytest.mark.parametrize(
    'bloq, classification',
    (
        (CSwap(10), 'swaps'),
        (HammingWeightPhasing(10, 1.11), 'rotations'),
        (Add(QInt(8)), 'arithmetic'),
        (QROM.build_from_data([4, 10, 11, 34]), 'data_loading'),
        (And(), 'multi_control_pauli'),
        # https://github.com/python/mypy/issues/5313
        (ReflectionUsingPrepare(PrepareIdentity.from_bitsizes((3, 3, 2))), 'reflection'),  # type: ignore[arg-type]
        (LessThanConstant(8, 3).adjoint(), 'arithmetic'),
    ),
)
def test_classify_bloq(bloq, classification):
    bloq_classification = _get_basic_bloq_classification()
    bloq_type = classify_bloq(bloq, bloq_classification)
    assert bloq_type == classification


def test_classify_bloq_counts_with_custom_bloq_classification():
    bloq_classification = {'qualtran.bloqs.basic_gates.swap': 'swaps'}
    test_bloq = TestBundleOfBloqs(((CSwap(10), 42), (Add(QInt(4)), 3)))
    classified_bloqs = classify_t_count_by_bloq_type(
        test_bloq, bloq_classification=bloq_classification
    )
    assert classified_bloqs == {'swaps': 42 * 10 * 7, 'other': 3 * 4 * (4 - 1)}
    assert get_cost_value(test_bloq, QECGatesCost()).total_t_count() == sum(
        classified_bloqs.values()
    )


def test_bloq_is_rotation():
    assert bloq_is_rotation(Rx(np.pi * 0.123))
    assert not bloq_is_rotation(Rx(np.pi / 2))
    assert bloq_is_rotation(Rx(np.pi * 0.123).controlled())
    assert bloq_is_rotation(Rx(np.pi / 2).controlled())
    assert not bloq_is_rotation(Rx(np.pi).controlled(ctrl_spec=CtrlSpec(cvs=(1, 1))))


def test_bloq_is_t_like():
    assert bloq_is_t_like(TGate())
    assert bloq_is_t_like(TGate().adjoint())
    assert bloq_is_t_like(TGate(is_adjoint=True))
    assert bloq_is_t_like(Rx(np.pi / 4))
    assert bloq_is_t_like(YPowGate(1.0 / 4))
    assert not bloq_is_t_like(Rz(np.pi / 2))
    assert not bloq_is_t_like(And())
