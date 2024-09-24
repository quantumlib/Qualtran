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

import qualtran.testing as qlt_testing
from qualtran.bloqs.factoring.ecc.ec_add import _ec_add, _ec_add_small, ECAdd


@pytest.mark.parametrize(
    'n,p,window_size,x,y,a,b,lam_r,result',
    [
        (32, 17, 1, 15, 13, 2, 10, 7, (15, 13, 8, 3, 7)),
        (32, 17, 4, 15, 13, 2, 10, 7, (15, 13, 8, 3, 7)),
        (32, 17, 8, 15, 13, 15, 13, 7, (15, 13, 2, 10, 7)),
    ],
)
def test_ec_add_classical(n, p, window_size, x, y, a, b, lam_r, result):
    bloq = ECAdd(n=n, mod=p, window_size=window_size)
    ret1 = bloq.call_classically(a=a, b=b, x=x, y=y, lam_r=lam_r)
    ret2 = bloq.decompose_bloq().call_classically(a=a, b=b, x=x, y=y, lam_r=lam_r)
    assert len(ret1) == len(ret2)
    for i in range(len(ret1)):
        np.testing.assert_array_equal(ret1[i], ret2[i])
        np.testing.assert_array_equal(ret1[i], result[i])


def test_ec_add(bloq_autotester):
    bloq_autotester(_ec_add)


def test_ec_add_small(bloq_autotester):
    bloq_autotester(_ec_add_small)


def test_notebook():
    qlt_testing.execute_notebook('ec_add')
