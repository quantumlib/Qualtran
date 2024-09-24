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

from qualtran.bloqs.factoring.ecc.ec_add_r import (
    _ec_add_r,
    _ec_add_r_small,
    _ec_window_add,
    _ec_window_add_r_small,
    ECAddR,
    ECWindowAddR,
)
from qualtran.bloqs.factoring.ecc.ec_point import ECPoint


@pytest.mark.parametrize(
    'n,p,curve_a,ctrl,x,y,a,b,result',
    [
        (8, 7, 3, 0, 0, 2, 1, 1, (0, 0, 2)),
        (8, 7, 3, 1, 0, 2, 1, 1, (1, 0, 5)),
        (16, 17, 0, 0, 15, 13, 2, 10, (0, 15, 13)),
        (16, 17, 0, 1, 15, 13, 2, 10, (1, 8, 3)),
    ],
)
def test_ec_add_r_classical(n, p, curve_a, ctrl, x, y, a, b, result):
    R = ECPoint(a, b, mod=p, curve_a=curve_a)
    bloq = ECAddR(n=n, R=R)
    ret1 = bloq.call_classically(ctrl=ctrl, x=x, y=y)
    ret2 = bloq.decompose_bloq().call_classically(ctrl=ctrl, x=x, y=y)
    assert len(ret1) == len(ret2)
    for i in range(len(ret1)):
        np.testing.assert_array_equal(ret1[i], ret2[i])
        np.testing.assert_array_equal(ret1[i], result[i])


@pytest.mark.parametrize(
    'n,p,curve_a,window_size,ctrl,x,y,a,b,result',
    [
        (16, 7, 3, 4, (0, 0, 0, 0), 0, 2, 1, 1, ((0, 0, 0, 0), 0, 2)),
        (16, 7, 3, 4, (0, 0, 0, 1), 0, 2, 1, 1, ((0, 0, 0, 1), 0, 5)),
        (16, 7, 3, 4, (0, 1, 0, 1), 0, 2, 1, 1, ((0, 1, 0, 1), 0, 2)),
        (32, 17, 0, 8, (0, 0, 0, 0, 0, 0, 0, 0), 15, 13, 2, 10, ((0, 0, 0, 0, 0, 0, 0, 0), 15, 13)),
        (32, 17, 0, 8, (0, 0, 0, 0, 0, 0, 0, 1), 15, 13, 2, 10, ((0, 0, 0, 0, 0, 0, 0, 1), 8, 3)),
        (32, 17, 0, 8, (0, 0, 0, 0, 1, 0, 0, 1), 15, 13, 2, 10, ((0, 0, 0, 0, 1, 0, 0, 1), 15, 13)),
    ],
)
def test_ec_window_add_r_classical(n, p, curve_a, window_size, ctrl, x, y, a, b, result):
    R = ECPoint(a, b, mod=p, curve_a=curve_a)
    bloq = ECWindowAddR(n=n, R=R, window_size=window_size)
    ret1 = bloq.call_classically(ctrl=ctrl, x=x, y=y)
    ret2 = bloq.decompose_bloq().call_classically(ctrl=ctrl, x=x, y=y)
    assert len(ret1) == len(ret2)
    for i in range(len(ret1)):
        np.testing.assert_array_equal(ret1[i], ret2[i])
        np.testing.assert_array_equal(ret1[i], result[i])


def test_ec_add_r(bloq_autotester):
    bloq_autotester(_ec_add_r)


def test_ec_add_r_small(bloq_autotester):
    bloq_autotester(_ec_add_r_small)


def test_ec_window_add(bloq_autotester):
    bloq_autotester(_ec_window_add)


def test_ec_window_add_r_small(bloq_autotester):
    bloq_autotester(_ec_window_add_r_small)
