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

from qualtran import QFxp, QInt, QUInt
from qualtran.bloqs.bookkeeping import Cast
from qualtran.bloqs.bookkeeping.cast import _cast
from qualtran.bloqs.for_testing import TestCastToFrom
from qualtran.simulation.tensor import cbloq_to_quimb


def test_cast(bloq_autotester):
    bloq_autotester(_cast)


def test_cast_tensor_contraction():
    bloq = TestCastToFrom()
    tn = cbloq_to_quimb(bloq.as_composite_bloq().flatten())
    assert tn.shape == (2,) * 4 * 4


def test_cast_classical_sim():
    qint = QUInt(8)
    qfxp = QFxp(8, 8)

    c = Cast(qint, qfxp)
    (y,) = c.call_classically(reg=7)
    assert y == int(y)
    assert qfxp.float_from_fixed_width_int(int(y)) == 7 / 2**8

    bloq = TestCastToFrom(bitsize=8)
    b_float = 2 / 2**8
    (a, b) = bloq.call_classically(a=7, b=qfxp.to_fixed_width_int(b_float))
    assert a == 7
    assert b == int(b)
    assert qfxp.float_from_fixed_width_int(int(b)) == 9 / 2**8

    c = Cast(qfxp, qint)
    val = 1.2
    val_as_int = qfxp.to_fixed_width_int(val)
    assert c.call_classically(reg=val_as_int) == (val_as_int,)  # type: ignore


def test_cast_unsiged_signed():
    c = Cast(QUInt(5), QInt(5))
    assert c.call_classically(reg=31) == (-1,)

    c = Cast(QInt(5), QUInt(5))
    assert c.call_classically(reg=-1) == (31,)
