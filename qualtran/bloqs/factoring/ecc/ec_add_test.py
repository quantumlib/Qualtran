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
from qualtran.bloqs.factoring.ecc.ec_add import _ec_add, _ec_add_small, _ECAddStepOne, _ECAddStepTwo, _ECAddStepThree, _ECAddStepFour, _ECAddStepFive, _ECAddStepSix, ECAdd


@pytest.mark.parametrize(
    'n,p,a,b,x,y,result',
    [
        (32, 17, 15, 13, 2, 10, (0, 0, 0, 0, 1, 15, 13, 2, 10)),
        (32, 17, 15, 13, 15, 13, (1, 0, 0, 0, 1, 15, 13, 15, 13)),
        (32, 17, 15, 13, 15, 4, (1, 1, 0, 0, 0, 15, 13, 15, 4)),
        (32, 17, 0, 0, 15, 4, (0, 0, 1, 0, 0, 0, 0, 15, 4)),
        (32, 17, 15, 13, 0, 0, (0, 0, 0, 1, 0, 15, 13, 0, 0)),
    ],
)
def test_ec_add_step_one_classical(n, p, a, b, x, y, result):
    bloq = _ECAddStepOne(n=n, mod=p)
    ret1 = bloq.call_classically(a=a, b=b, x=x, y=y)
    ret2 = bloq.decompose_bloq().call_classically(a=a, b=b, x=x, y=y)
    np.testing.assert_array_equal(ret1, ret2)
    np.testing.assert_array_equal(ret1, result)


@pytest.mark.parametrize(
    'n,p,window_size,f1,ctrl,a,b,x,y,lam_r,result',
    [
        (32, 17, 1, 0, 1, 15, 13, 2, 10, 7, (0, 1, 15, 13, 4, 14, 12, 7)),
        (32, 17, 4, 1, 1, 15, 13, 15, 13, 7, (0, 1, 15, 13, 0, 0, 7, 7)),
        (32, 17, 8, 1, 0, 15, 13, 15, 4, 7, (1, 0, 15, 13, 0, 4, 0, 7)),
        (32, 17, 16, 0, 0, 0, 0, 15, 4, 1, (0, 0, 0, 0, 15, 4, 0, 1)),
    ],
)
def test_ec_add_step_two_classical(n, p, window_size, f1, ctrl, a, b, x, y, lam_r, result):
    bloq = _ECAddStepTwo(n=n, mod=p, window_size=window_size)
    ret1 = bloq.call_classically(f1=f1, ctrl=ctrl, a=a, b=b, x=x, y=y, lam_r=lam_r)
    ret2 = bloq.decompose_bloq().call_classically(f1=f1, ctrl=ctrl, a=a, b=b, x=x, y=y, lam_r=lam_r)
    np.testing.assert_array_equal(ret1, ret2)
    np.testing.assert_array_equal(ret1, result)


@pytest.mark.parametrize(
    'n,p,window_size,ctrl,a,b,x,y,lam,result',
    [
        (32, 17, 1, 1, 15, 13, 4, 14, 12, (1, 15, 13, 15, 0, 12)),
        (32, 17, 4, 1, 15, 13, 0, 0, 7, (1, 15, 13, 11, 0, 7)),
        (32, 17, 8, 0, 15, 13, 0, 4, 0, (0, 15, 13, 0, 4, 0)),
        (32, 17, 16, 0, 0, 0, 15, 4, 0, (0, 0, 0, 15, 4, 0)),
    ],
)
def test_ec_add_step_three_classical(n, p, window_size, ctrl, a, b, x, y, lam, result):
    bloq = _ECAddStepThree(n=n, mod=p, window_size=window_size)
    ret1 = bloq.call_classically(ctrl=ctrl, a=a, b=b, x=x, y=y, lam=lam)
    ret2 = bloq.decompose_bloq().call_classically(ctrl=ctrl, a=a, b=b, x=x, y=y, lam=lam)
    np.testing.assert_array_equal(ret1, ret2)
    np.testing.assert_array_equal(ret1, result)


@pytest.mark.parametrize(
    'n,p,window_size,a,b,x,y,lam_r,result',
    [
        (32, 17, 1, 15, 13, 2, 10, 7, (15, 13, 8, 3, 7)),
        (32, 17, 4, 15, 13, 2, 10, 7, (15, 13, 8, 3, 7)),
        (32, 17, 8, 15, 13, 15, 13, 7, (15, 13, 2, 10, 7)),
    ],
)
def test_ec_add_classical(n, p, window_size, a, b, x, y, lam_r, result):
    bloq = ECAdd(n=n, mod=p, window_size=window_size)
    ret1 = bloq.call_classically(a=a, b=b, x=x, y=y, lam_r=lam_r)
    ret2 = bloq.decompose_bloq().call_classically(a=a, b=b, x=x, y=y, lam_r=lam_r)
    np.testing.assert_array_equal(ret1, ret2)
    np.testing.assert_array_equal(ret1, result)


def test_ec_add(bloq_autotester):
    bloq_autotester(_ec_add)


def test_ec_add_small(bloq_autotester):
    bloq_autotester(_ec_add_small)


def test_notebook():
    qlt_testing.execute_notebook('ec_add')
