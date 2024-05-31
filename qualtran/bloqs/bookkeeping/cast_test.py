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
import subprocess

from qualtran import QFxp, QInt
from qualtran.bloqs.bookkeeping import Cast
from qualtran.bloqs.bookkeeping.cast import _cast
from qualtran.bloqs.for_testing import TestCastToFrom
from qualtran.simulation.tensor import cbloq_to_quimb


def test_cast(bloq_autotester):
    bloq_autotester(_cast)


def test_cast_tensor_contraction():
    bloq = TestCastToFrom()
    tn, _ = cbloq_to_quimb(bloq.decompose_bloq())
    assert len(tn.tensors) == 3
    assert tn.shape == (2**4,) * 4


def test_cast_classical_sim():
    c = Cast(QInt(8), QFxp(8, 8))
    (y,) = c.call_classically(reg=7)
    assert y == 7
    bloq = TestCastToFrom()
    (a, b) = bloq.call_classically(a=7, b=2)
    assert a == 7
    assert b == 9


def test_no_circular_import():
    subprocess.check_call(['python', '-c', 'from qualtran.bloqs.bookkeeping import cast'])
