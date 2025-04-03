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
from qualtran import QMontgomeryUInt, QUInt
from qualtran.bloqs.cryptography.ecc.ec_add_r import (
    _ec_add_r,
    _ec_add_r_small,
    _ec_window_add_r_small,
    ECWindowAddR,
)
from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join

from .ec_add_r import ECWindowAddR
from .ec_point import ECPoint


@pytest.mark.parametrize('bloq', [_ec_add_r, _ec_add_r_small, _ec_window_add_r_small])
def test_ec_add_r(bloq_autotester, bloq):
    bloq_autotester(bloq)


@pytest.mark.parametrize('a,b', [(15, 13), (0, 0)])
@pytest.mark.parametrize(
    ['n', 'window_size'],
    [
        (n, window_size)
        for n in range(5, 8)
        for window_size in range(1, n + 1)
        if n % window_size == 0
    ],
)
def test_ec_window_add_r_bloq_counts(n, window_size, a, b):
    p = 17
    R = ECPoint(a, b, mod=p)
    bloq = ECWindowAddR(n=n, R=R, add_window_size=window_size)
    qlt_testing.assert_equivalent_bloq_counts(bloq, [ignore_alloc_free, ignore_split_join])


@pytest.mark.parametrize(
    ['n', 'm'], [(n, m) for n in range(4, 5) for m in range(1, n + 1) if n % m == 0]
)
@pytest.mark.parametrize('a,b', [(15, 13), (0, 0)])
@pytest.mark.parametrize('x,y', [(15, 13), (5, 8)])
@pytest.mark.parametrize('ctrl', [0, 1, 5])
def test_ec_window_add_r_classical(n, m, ctrl, x, y, a, b):
    p = 17
    R = ECPoint(a, b, mod=p)
    x = QMontgomeryUInt(n, p).uint_to_montgomery(x)
    y = QMontgomeryUInt(n, p).uint_to_montgomery(y)
    ctrl = np.array(QUInt(m).to_bits(ctrl % (2**m)))
    bloq = ECWindowAddR(n=n, R=R, add_window_size=m, mul_window_size=m)
    ret1 = bloq.call_classically(ctrl=ctrl, x=x, y=y)
    ret2 = bloq.decompose_bloq().call_classically(ctrl=ctrl, x=x, y=y)
    for i, ret1_i in enumerate(ret1):
        np.testing.assert_array_equal(ret1_i, ret2[i])


@pytest.mark.slow
@pytest.mark.parametrize(
    ['n', 'm'], [(n, m) for n in range(7, 9) for m in range(1, n + 1) if n % m == 0]
)
@pytest.mark.parametrize('a,b', [(15, 13), (0, 0)])
@pytest.mark.parametrize('x,y', [(15, 13), (5, 8)])
@pytest.mark.parametrize('ctrl', [0, 1, 5, 8])
def test_ec_window_add_r_classical_slow(n, m, ctrl, x, y, a, b):
    p = 17
    R = ECPoint(a, b, mod=p)
    x = QMontgomeryUInt(n, p).uint_to_montgomery(x)
    y = QMontgomeryUInt(n, p).uint_to_montgomery(y)
    ctrl = np.array(QUInt(m).to_bits(ctrl % (2**m)))
    bloq = ECWindowAddR(n=n, R=R, add_window_size=m, mul_window_size=m)
    ret1 = bloq.call_classically(ctrl=ctrl, x=x, y=y)
    ret2 = bloq.decompose_bloq().call_classically(ctrl=ctrl, x=x, y=y)
    for i, ret1_i in enumerate(ret1):
        np.testing.assert_array_equal(ret1_i, ret2[i])
