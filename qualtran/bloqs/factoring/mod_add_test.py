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

import numpy as np
import pytest

from qualtran.bloqs.factoring.mod_add import CtrlModAddK, CtrlScaleModAdd, MontgomeryModAdd
from qualtran.testing import assert_valid_bloq_decomposition


def test_ctrl_scale_mod_add():
    bloq = CtrlScaleModAdd(k=123, mod=13 * 17, bitsize=8)
    assert bloq.short_name() == 'y += x*123 % 221'

    counts = bloq.bloq_counts()
    ((bloq, n),) = counts.items()
    assert n == 8


def test_ctrl_mod_add_k():
    bloq = CtrlModAddK(k=123, mod=13 * 17, bitsize=8)
    assert bloq.short_name() == 'x += 123 % 221'

    counts = bloq.bloq_counts()
    ((bloq, n),) = counts.items()
    assert n == 5


@pytest.mark.parametrize('bitsize,p', [(1, 1), (2, 3), (5, 8)])
def test_montgomery_mod_add_decomp(bitsize, p):
    bloq = MontgomeryModAdd(bitsize=bitsize, p=p)
    assert_valid_bloq_decomposition(bloq)


# TODO: write tests for signed integer comparison
# https://github.com/quantumlib/Qualtran/issues/606
@pytest.mark.parametrize(
    'bitsize,p,x,y,result',
    [
        (3, 3, 1, 4, 2),
        (4, 6, 2, 7, 3),
        (5, 11, 8, 9, 6),
        (6, 4, 2, 4, 2),
        (7, 20, 20, 11, 11),
        (8, 80, 50, 10, 60),
    ],
)
def test_classical_montgomery_mod_add(bitsize, p, x, y, result):
    bloq = MontgomeryModAdd(bitsize=bitsize, p=p)
    cbloq = bloq.decompose_bloq()
    bloq_classical = bloq.call_classically(x=x, y=y)
    cbloq_classical = cbloq.call_classically(x=x, y=y)

    assert len(bloq_classical) == len(cbloq_classical)
    for i in range(len(bloq_classical)):
        np.testing.assert_array_equal(bloq_classical[i], cbloq_classical[i])

    assert bloq_classical[-1] == result
