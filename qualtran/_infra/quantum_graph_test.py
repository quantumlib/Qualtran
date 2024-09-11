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

from qualtran import BloqInstance, DanglingT, LeftDangle, QAny, Register, RightDangle, Side, Soquet
from qualtran.bloqs.for_testing import TestAtom, TestTwoBitOp


def test_dangling():
    assert LeftDangle is LeftDangle
    assert RightDangle is RightDangle
    assert LeftDangle is not RightDangle
    assert RightDangle is not LeftDangle

    assert isinstance(LeftDangle, DanglingT)
    assert isinstance(RightDangle, DanglingT)

    assert LeftDangle == LeftDangle
    assert RightDangle == RightDangle
    assert LeftDangle != RightDangle

    with pytest.raises(ValueError, match='Do not instantiate.*'):
        my_new_dangle = DanglingT('hi mom')


def test_dangling_hash():
    assert hash(LeftDangle) != hash(RightDangle)
    my_d = {LeftDangle: 'left', RightDangle: 'right'}
    assert my_d[LeftDangle] == 'left'
    assert my_d[RightDangle] == 'right'


def test_soquet():
    soq = Soquet(BloqInstance(TestTwoBitOp(), i=0), Register('x', QAny(10)))
    assert soq.reg.side is Side.THRU
    assert soq.idx == ()
    assert soq.pretty() == 'x'


def test_soquet_idxed():
    binst = BloqInstance(TestTwoBitOp(), i=0)
    reg = Register('y', QAny(10), shape=(10, 2))

    with pytest.raises(ValueError, match=r'Bad index.*'):
        _ = Soquet(binst, reg)

    with pytest.raises(ValueError, match=r'Bad index.*'):
        _ = Soquet(binst, reg, idx=(5,))

    soq = Soquet(binst, reg, idx=(5, 0))
    assert soq.pretty() == 'y[5, 0]'

    with pytest.raises(ValueError, match=r'Bad index.*'):
        _ = Soquet(binst, reg, idx=(5,))


def test_bloq_instance():
    binst_a = BloqInstance(TestAtom(), i=1)
    binst_b = BloqInstance(TestAtom(), i=1)
    assert binst_a == binst_b
    assert str(binst_a) == 'TestAtom<1>'
