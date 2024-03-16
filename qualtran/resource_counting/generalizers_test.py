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
import cirq

from qualtran import QAny
from qualtran.bloqs.basic_gates import CNOT, Rx, TwoBitSwap
from qualtran.bloqs.mcmt.and_bloq import And, MultiAnd
from qualtran.bloqs.util_bloqs import Allocate, Free, Join, Split
from qualtran.cirq_interop import CirqGateAsBloq
from qualtran.resource_counting.bloq_counts import _make_composite_generalizer
from qualtran.resource_counting.generalizers import (
    cirq_to_bloqs,
    CV,
    generalize_cvs,
    generalize_rotation_angle,
    ignore_alloc_free,
    ignore_cliffords,
    ignore_split_join,
    PHI,
)

_BLOQS_TO_FILTER = [
    CNOT(),
    CirqGateAsBloq(cirq.CNOT),
    Split(QAny(bitsize=5)),
    Join(QAny(bitsize=5)),
    TwoBitSwap(),
    And(0, 0),
    MultiAnd((1, 0, 1, 0)),
    Rx(0.123),
    Allocate(QAny(bitsize=5)),
    Free(QAny(bitsize=5)),
]


def test_ignore_split_join():
    bloqs = [ignore_split_join(b) for b in _BLOQS_TO_FILTER]
    assert bloqs == [
        CNOT(),
        CirqGateAsBloq(cirq.CNOT),
        None,  # Split(QAny(bitsize=5))
        None,  # Join(QAny(bitsize=5))
        TwoBitSwap(),
        And(0, 0),
        MultiAnd((1, 0, 1, 0)),
        Rx(0.123),
        Allocate(QAny(bitsize=5)),
        Free(QAny(bitsize=5)),
    ]


def test_ignore_alloc_free():
    bloqs = [ignore_alloc_free(b) for b in _BLOQS_TO_FILTER]
    assert bloqs == [
        CNOT(),
        CirqGateAsBloq(cirq.CNOT),
        Split(QAny(bitsize=5)),
        Join(QAny(bitsize=5)),
        TwoBitSwap(),
        And(0, 0),
        MultiAnd((1, 0, 1, 0)),
        Rx(0.123),
        None,  # Allocate(QAny(bitsize=5))
        None,  # Free(QAny(bitsize=5))
    ]


def test_generalize_rotation_angle():
    bloqs = [generalize_rotation_angle(b) for b in _BLOQS_TO_FILTER]
    assert bloqs == [
        CNOT(),
        CirqGateAsBloq(cirq.CNOT),
        Split(QAny(bitsize=5)),
        Join(QAny(bitsize=5)),
        TwoBitSwap(),
        And(0, 0),
        MultiAnd((1, 0, 1, 0)),
        Rx(PHI),  # this one is generalized
        Allocate(QAny(bitsize=5)),
        Free(QAny(bitsize=5)),
    ]


def test_generalize_cvs():
    bloqs = [generalize_cvs(b) for b in _BLOQS_TO_FILTER]
    assert bloqs == [
        CNOT(),
        CirqGateAsBloq(cirq.CNOT),
        Split(QAny(bitsize=5)),
        Join(QAny(bitsize=5)),
        TwoBitSwap(),
        And(CV, CV),  # changed
        MultiAnd((CV,) * 4),  # changed
        Rx(0.123),
        Allocate(QAny(bitsize=5)),
        Free(QAny(bitsize=5)),
    ]


def test_ignore_cliffords():
    bloqs = [ignore_cliffords(b) for b in _BLOQS_TO_FILTER]
    assert bloqs == [
        None,  # CNOT(),
        CirqGateAsBloq(cirq.CNOT),
        Split(QAny(bitsize=5)),
        Join(QAny(bitsize=5)),
        None,  # TwoBitSwap(),
        And(0, 0),
        MultiAnd((1, 0, 1, 0)),
        Rx(0.123),
        Allocate(QAny(bitsize=5)),
        Free(QAny(bitsize=5)),
    ]


def test_ignore_cliffords_with_cirq():
    gg = _make_composite_generalizer(cirq_to_bloqs, ignore_cliffords)
    bloqs = [gg(b) for b in _BLOQS_TO_FILTER]
    assert bloqs == [
        None,  # CNOT(),
        None,  # CirqGateAsBloq(cirq.CNOT),
        Split(QAny(bitsize=5)),
        Join(QAny(bitsize=5)),
        None,  # TwoBitSwap(),
        And(0, 0),
        MultiAnd((1, 0, 1, 0)),
        Rx(0.123),
        Allocate(QAny(bitsize=5)),
        Free(QAny(bitsize=5)),
    ]


def test_many_generalizers():
    gg = _make_composite_generalizer(
        cirq_to_bloqs,
        ignore_cliffords,
        ignore_alloc_free,
        ignore_split_join,
        generalize_cvs,
        generalize_rotation_angle,
    )
    bloqs = [gg(b) for b in _BLOQS_TO_FILTER]
    bloqs = [b for b in bloqs if b is not None]
    assert bloqs == [
        # CNOT(),
        # CirqGateAsBloq(cirq.CNOT),
        # Split(QAny(n=5)),
        # Join(QAny(n=5)),
        # TwoBitSwap(),
        And(CV, CV),
        MultiAnd((CV,) * 4),
        Rx(PHI),
        # Allocate(QAny(n=5)),
        # Free(QAny(n=5)),
    ]
