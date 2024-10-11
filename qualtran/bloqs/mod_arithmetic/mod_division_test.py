#  Copyright 2024 Google LLC
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
import math
from qualtran import (
    QMontgomeryUInt
)

from qualtran.resource_counting import get_cost_value, QECGatesCost
from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join
from qualtran.bloqs.mod_arithmetic import KaliskiModInverse
import qualtran.testing as qlt_testing


@pytest.mark.parametrize('bitsize', [5, 6])
@pytest.mark.parametrize('mod', [3, 5, 7, 11, 13, 15])
def test_kaliski_mod_inverse_classical_action(bitsize, mod):
    blq = KaliskiModInverse(bitsize, mod)
    cblq = blq.decompose_bloq()
    p2 = pow(2, bitsize, mod)
    for x in range(1, mod):
        if math.gcd(x, mod) != 1: continue
        x_montgomery = (x * p2)%mod
        inv_x = pow(x, -1, mod)
        inv_x_montgomery = (inv_x * p2) % mod
        res = blq.call_classically(u=mod, v=x_montgomery, r=0, s=1)
        assert res == cblq.call_classically(u=mod, v=x_montgomery, r=0, s=1)
        u, v, r, s = res[:4]
    
        # Invariants of the Kaliski algorithm.
        assert u == 1
        assert v == 0
        assert s == mod
        assert r == inv_x_montgomery

@pytest.mark.parametrize('bitsize', [5, 6])
@pytest.mark.parametrize('mod', [3, 5, 7, 11, 13, 15])
def test_kaliski_mod_inverse_decomposition(bitsize, mod):
    b = KaliskiModInverse(bitsize, mod)
    qlt_testing.assert_valid_bloq_decomposition(b)

@pytest.mark.parametrize('bitsize', [5, 6])
@pytest.mark.parametrize('mod', [3, 5, 7, 11, 13, 15])
def test_kaliski_mod_bloq_counts(bitsize, mod):
    b = KaliskiModInverse(bitsize, mod)
    qlt_testing.assert_equivalent_bloq_counts(b, [ignore_alloc_free, ignore_split_join])