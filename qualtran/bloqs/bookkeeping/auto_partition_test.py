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

import pytest
from attrs import evolve

from qualtran import Controlled, CtrlSpec, QAny, Register
from qualtran.bloqs.basic_gates import Swap
from qualtran.bloqs.bookkeeping.auto_partition import (
    _auto_partition,
    _auto_partition_unused,
    AutoPartition,
    Unused,
)


def test_auto_partition(bloq_autotester):
    bloq_autotester(_auto_partition)
    bloq_autotester(_auto_partition_unused)


def test_auto_partition_input():
    from qualtran import QAny, QBit, Register, Side

    bloq = _auto_partition()

    assert tuple(bloq.signature.lefts()) == (
        Register('x', dtype=QAny(2)),
        Register('y', dtype=QAny(1)),
    )

    assert tuple(bloq.signature.rights()) == (
        Register('x', dtype=QAny(2)),
        Register('y', dtype=QAny(1)),
    )

    x, y = bloq.call_classically(x=0b11, y=0b0)
    assert x == 0b10
    assert y == 0b1

    x, y = bloq.call_classically(x=0b01, y=0b0)
    assert x == 0b01
    assert y == 0b0

    bloq = evolve(bloq, left_only=True)

    assert tuple(bloq.signature.lefts()) == (
        Register('x', dtype=QAny(2), side=Side.LEFT),
        Register('y', dtype=QAny(1), side=Side.LEFT),
    )

    assert tuple(bloq.signature.rights()) == (
        Register('ctrl', dtype=QBit(), side=Side.RIGHT),
        Register('x', dtype=QBit(), side=Side.RIGHT),
        Register('y', dtype=QBit(), side=Side.RIGHT),
    )

    ctrl, x, y = bloq.call_classically(x=0b11, y=0b0)
    assert ctrl == 0b1
    assert x == 0b0
    assert y == 0b1

    ctrl, x, y = bloq.call_classically(x=0b01, y=0b0)
    assert ctrl == 0b0
    assert x == 0b1
    assert y == 0b0


def test_auto_partition_valid():
    from qualtran import Controlled, CtrlSpec, QAny, QUInt, Register
    from qualtran.bloqs.basic_gates import Swap, XGate

    with pytest.raises(ValueError):
        bloq = Controlled(Swap(3), CtrlSpec(qdtypes=QUInt(4), cvs=0b0110))
        bloq = AutoPartition(
            bloq, [(Register('a', QAny(3)), ['y']), (Register('b', QAny(3)), ['x'])]
        )

    with pytest.raises(ValueError):
        bloq = AutoPartition(
            XGate(),
            [(Register('a', QAny(1)), ['q']), (Register('b', QAny(1)), [Unused(1)])],
            left_only=True,
        )


def test_auto_partition_big():
    from qualtran import Controlled, CtrlSpec, QAny, QUInt, Register, Side
    from qualtran.bloqs.basic_gates import Swap

    bloq = Controlled(Swap(3), CtrlSpec(qdtypes=QUInt(4), cvs=0b0110))
    bloq = AutoPartition(
        bloq, [(Register('a', QAny(7)), ['y', 'ctrl']), (Register('b', QAny(3)), ['x'])]
    )

    assert tuple(bloq.signature.lefts()) == (
        Register('a', dtype=QAny(7)),
        Register('b', dtype=QAny(3)),
    )

    assert tuple(bloq.signature.rights()) == (
        Register('a', dtype=QAny(7)),
        Register('b', dtype=QAny(3)),
    )

    a, b = bloq.call_classically(a=0b1010110, b=0b010)
    assert a == 0b0100110
    assert b == 0b101

    a, b = bloq.call_classically(a=0b1010111, b=0b010)
    assert a == 0b1010111
    assert b == 0b010

    bloq = evolve(bloq, left_only=True)

    assert tuple(bloq.signature.lefts()) == (
        Register('a', dtype=QAny(7), side=Side.LEFT),
        Register('b', dtype=QAny(3), side=Side.LEFT),
    )

    assert tuple(bloq.signature.rights()) == (
        Register('ctrl', dtype=QUInt(4), side=Side.RIGHT),
        Register('x', dtype=QAny(3), side=Side.RIGHT),
        Register('y', dtype=QAny(3), side=Side.RIGHT),
    )

    ctrl, x, y = bloq.call_classically(a=0b1010110, b=0b010)
    assert ctrl == 0b0110
    assert x == 0b101
    assert y == 0b010

    ctrl, x, y = bloq.call_classically(a=0b1010111, b=0b010)
    assert ctrl == 0b0111
    assert x == 0b010
    assert y == 0b101


def test_auto_partition_unused():
    from qualtran import QAny, Register

    bloq = _auto_partition_unused()

    assert tuple(bloq.signature.lefts()) == (
        Register('x', dtype=QAny(3)),
        Register('y', dtype=QAny(1)),
        Register('z', dtype=QAny(2)),
    )

    assert tuple(bloq.signature.rights()) == tuple(bloq.signature.lefts())

    x, y, z = bloq.call_classically(x=0b111, y=0b0, z=0b10)
    assert x == 0b101
    assert y == 0b1
    assert z == 0b10

    x, y, z = bloq.call_classically(x=0b010, y=0b0, z=0b01)
    assert x == 0b010
    assert y == 0b0
    assert z == 0b01

    with pytest.raises(ValueError):
        _ = evolve(bloq, left_only=True)


def test_auto_partition_unused_index():
    bloq = Controlled(Swap(1), CtrlSpec())
    bloq = AutoPartition(
        bloq,
        [
            (Register('x', QAny(3)), [Unused(1), 'ctrl', 'x']),
            (Register('y', QAny(1)), ['y']),
            (Register('z', QAny(2)), [Unused(2)]),
        ],
    )

    assert tuple(bloq.signature.lefts()) == (
        Register('x', dtype=QAny(3)),
        Register('y', dtype=QAny(1)),
        Register('z', dtype=QAny(2)),
    )

    assert tuple(bloq.signature.rights()) == tuple(bloq.signature.lefts())

    x, y, z = bloq.call_classically(x=0b111, y=0b0, z=0b10)
    assert x == 0b110
    assert y == 0b1
    assert z == 0b10

    x, y, z = bloq.call_classically(x=0b001, y=0b0, z=0b01)
    assert x == 0b001
    assert y == 0b0
    assert z == 0b01

    with pytest.raises(ValueError):
        _ = evolve(bloq, left_only=True)
