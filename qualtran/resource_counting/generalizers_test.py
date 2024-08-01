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

from qualtran import Adjoint, QAny, Register
from qualtran.bloqs.basic_gates import CNOT, CSwap, Rx, TwoBitSwap
from qualtran.bloqs.bookkeeping import Allocate, AutoPartition, Free, Join, Partition, Split
from qualtran.bloqs.mcmt.and_bloq import And, MultiAnd
from qualtran.bloqs.swap_network import CSwapApprox
from qualtran.cirq_interop import CirqGateAsBloq
from qualtran.resource_counting._generalization import _make_composite_generalizer
from qualtran.resource_counting.generalizers import (
    cirq_to_bloqs,
    CV,
    generalize_cswap_approx,
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
    CSwapApprox(bitsize=44),
    CSwapApprox(bitsize=3).adjoint(),
    Split(QAny(bitsize=5)),
    Join(QAny(bitsize=5)),
    TwoBitSwap(),
    And(0, 0),
    MultiAnd((1, 0, 1, 0)),
    Rx(0.123),
    Allocate(QAny(bitsize=5)),
    Free(QAny(bitsize=5)),
    Adjoint(TwoBitSwap()),
    Partition(5, (Register('x', QAny(2)), Register('y', QAny(3)))),
    CirqGateAsBloq(cirq.S),
    AutoPartition(
        AutoPartition(Rx(0.123), [(Register('q', QAny(1)), ['q'])]),
        [(Register('q', QAny(1)), ['q'])],
    ),
]


def test_ignore_split_join():
    bloqs = [ignore_split_join(b) for b in _BLOQS_TO_FILTER]
    assert bloqs == [
        CNOT(),
        CirqGateAsBloq(cirq.CNOT),
        CSwapApprox(bitsize=44),
        CSwapApprox(bitsize=3).adjoint(),
        None,  # Split(QAny(bitsize=5))
        None,  # Join(QAny(bitsize=5))
        TwoBitSwap(),
        And(0, 0),
        MultiAnd((1, 0, 1, 0)),
        Rx(0.123),
        Allocate(QAny(bitsize=5)),
        Free(QAny(bitsize=5)),
        Adjoint(TwoBitSwap()),
        None,  # Partition(5, (Register('x', QAny(2)), Register('y', QAny(3))))
        CirqGateAsBloq(cirq.S),
        Rx(0.123),
    ]


def test_ignore_alloc_free():
    bloqs = [ignore_alloc_free(b) for b in _BLOQS_TO_FILTER]
    assert bloqs == [
        CNOT(),
        CirqGateAsBloq(cirq.CNOT),
        CSwapApprox(bitsize=44),
        CSwapApprox(bitsize=3).adjoint(),
        Split(QAny(bitsize=5)),
        Join(QAny(bitsize=5)),
        TwoBitSwap(),
        And(0, 0),
        MultiAnd((1, 0, 1, 0)),
        Rx(0.123),
        None,  # Allocate(QAny(bitsize=5))
        None,  # Free(QAny(bitsize=5))
        Adjoint(TwoBitSwap()),
        Partition(5, (Register('x', QAny(2)), Register('y', QAny(3)))),
        CirqGateAsBloq(cirq.S),
        AutoPartition(
            AutoPartition(Rx(0.123), [(Register('q', QAny(1)), ['q'])]),
            [(Register('q', QAny(1)), ['q'])],
        ),
    ]


def test_generalize_rotation_angle():
    bloqs = [generalize_rotation_angle(b) for b in _BLOQS_TO_FILTER]
    assert bloqs == [
        CNOT(),
        CirqGateAsBloq(cirq.CNOT),
        CSwapApprox(bitsize=44),
        CSwapApprox(bitsize=3).adjoint(),
        Split(QAny(bitsize=5)),
        Join(QAny(bitsize=5)),
        TwoBitSwap(),
        And(0, 0),
        MultiAnd((1, 0, 1, 0)),
        Rx(PHI),  # this one is generalized
        Allocate(QAny(bitsize=5)),
        Free(QAny(bitsize=5)),
        Adjoint(TwoBitSwap()),
        Partition(5, (Register('x', QAny(2)), Register('y', QAny(3)))),
        CirqGateAsBloq(cirq.S),
        AutoPartition(
            AutoPartition(Rx(PHI), [(Register('q', QAny(1)), ['q'])]),
            [(Register('q', QAny(1)), ['q'])],
        ),
    ]


def test_generalize_cswap_approx():
    bloqs = [generalize_cswap_approx(b) for b in _BLOQS_TO_FILTER]
    assert bloqs == [
        CNOT(),
        CirqGateAsBloq(cirq.CNOT),
        CSwap(bitsize=44),
        CSwap(bitsize=3).adjoint(),
        Split(QAny(bitsize=5)),
        Join(QAny(bitsize=5)),
        TwoBitSwap(),
        And(0, 0),
        MultiAnd((1, 0, 1, 0)),
        Rx(0.123),
        Allocate(QAny(bitsize=5)),
        Free(QAny(bitsize=5)),
        Adjoint(TwoBitSwap()),
        Partition(5, (Register('x', QAny(2)), Register('y', QAny(3)))),
        CirqGateAsBloq(cirq.S),
        AutoPartition(
            AutoPartition(Rx(0.123), [(Register('q', QAny(1)), ['q'])]),
            [(Register('q', QAny(1)), ['q'])],
        ),
    ]


def test_generalize_cvs():
    bloqs = [generalize_cvs(b) for b in _BLOQS_TO_FILTER]
    assert bloqs == [
        CNOT(),
        CirqGateAsBloq(cirq.CNOT),
        CSwapApprox(bitsize=44),
        CSwapApprox(bitsize=3).adjoint(),
        Split(QAny(bitsize=5)),
        Join(QAny(bitsize=5)),
        TwoBitSwap(),
        And(CV, CV),  # changed
        MultiAnd((1,) * 4),  # changed
        Rx(0.123),
        Allocate(QAny(bitsize=5)),
        Free(QAny(bitsize=5)),
        Adjoint(TwoBitSwap()),
        Partition(5, (Register('x', QAny(2)), Register('y', QAny(3)))),
        CirqGateAsBloq(cirq.S),
        AutoPartition(
            AutoPartition(Rx(0.123), [(Register('q', QAny(1)), ['q'])]),
            [(Register('q', QAny(1)), ['q'])],
        ),
    ]


def test_ignore_cliffords():
    bloqs = [ignore_cliffords(b) for b in _BLOQS_TO_FILTER]
    assert bloqs == [
        None,  # CNOT(),
        CirqGateAsBloq(cirq.CNOT),
        CSwapApprox(bitsize=44),
        CSwapApprox(bitsize=3).adjoint(),
        Split(QAny(bitsize=5)),
        Join(QAny(bitsize=5)),
        None,  # TwoBitSwap(),
        And(0, 0),
        MultiAnd((1, 0, 1, 0)),
        Rx(0.123),
        Allocate(QAny(bitsize=5)),
        Free(QAny(bitsize=5)),
        None,  # Adjoint(TwoBitSwap()),
        Partition(5, (Register('x', QAny(2)), Register('y', QAny(3)))),
        CirqGateAsBloq(cirq.S),
        AutoPartition(
            AutoPartition(Rx(0.123), [(Register('q', QAny(1)), ['q'])]),
            [(Register('q', QAny(1)), ['q'])],
        ),
    ]


def test_ignore_cliffords_with_cirq():
    gg = _make_composite_generalizer(cirq_to_bloqs, ignore_cliffords)
    bloqs = [gg(b) for b in _BLOQS_TO_FILTER]
    assert bloqs == [
        None,  # CNOT(),
        None,  # CirqGateAsBloq(cirq.CNOT),
        CSwapApprox(bitsize=44),
        CSwapApprox(bitsize=3).adjoint(),
        Split(QAny(bitsize=5)),
        Join(QAny(bitsize=5)),
        None,  # TwoBitSwap(),
        And(0, 0),
        MultiAnd((1, 0, 1, 0)),
        Rx(0.123),
        Allocate(QAny(bitsize=5)),
        Free(QAny(bitsize=5)),
        None,  # Adjoint(TwoBitSwap()),
        Partition(5, (Register('x', QAny(2)), Register('y', QAny(3)))),
        None,  # cirq.S,
        AutoPartition(
            AutoPartition(Rx(0.123), [(Register('q', QAny(1)), ['q'])]),
            [(Register('q', QAny(1)), ['q'])],
        ),
    ]


def test_many_generalizers():
    gg = _make_composite_generalizer(
        cirq_to_bloqs,
        ignore_cliffords,
        ignore_alloc_free,
        ignore_split_join,
        generalize_cvs,
        generalize_rotation_angle,
        generalize_cswap_approx,
    )
    bloqs = [gg(b) for b in _BLOQS_TO_FILTER]
    bloqs = [b for b in bloqs if b is not None]
    assert bloqs == [
        # CNOT(),
        # CirqGateAsBloq(cirq.CNOT),
        CSwap(bitsize=44),
        CSwap(bitsize=3).adjoint(),
        # Split(QAny(n=5)),
        # Join(QAny(n=5)),
        # TwoBitSwap(),
        And(CV, CV),
        MultiAnd((1,) * 4),  # changed
        Rx(PHI),
        # Allocate(QAny(n=5)),
        # Free(QAny(n=5)),
        # Adjoint(TwoBitSwap()),
        # Partition(5, (Register('x', QAny(2)), Register('y', QAny(3))))
        # cirq.S,
        Rx(PHI),
    ]
