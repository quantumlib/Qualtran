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
import sympy
from galois import GF

from qualtran.bloqs.gf_arithmetic.gf2_inverse import (
    _gf2_inverse_symbolic,
    _gf16_inverse,
    GF2Inverse,
)
from qualtran.resource_counting import get_cost_value, QECGatesCost, QubitCount
from qualtran.resource_counting.generalizers import ignore_alloc_free, ignore_split_join
from qualtran.symbolics import ceil, log2
from qualtran.testing import assert_consistent_classical_action, assert_equivalent_bloq_counts


def test_gf16_inverse(bloq_autotester):
    bloq_autotester(_gf16_inverse)


def test_gf2_inverse_symbolic(bloq_autotester):
    bloq_autotester(_gf2_inverse_symbolic)


def test_gf2_inverse_symbolic_toffoli_complexity():
    bloq = _gf2_inverse_symbolic.make()
    m = bloq.bitsize
    expected_expr = m**2 * (2 * ceil(log2(m)) - 1)
    assert get_cost_value(bloq, QECGatesCost()).total_toffoli_only() - expected_expr == 0
    expected_expr = m * (3 * ceil(log2(m)) + 2)
    assert sympy.simplify(get_cost_value(bloq, QubitCount()) - expected_expr) == 0


def test_gf2_inverse_classical_sim_quick():
    m = 1
    bloq = GF2Inverse(m)
    GFM = GF(2**m)
    assert_consistent_classical_action(bloq, x=GFM.elements[1:])


@pytest.mark.slow
@pytest.mark.parametrize('m', [2, 3, 4, 5])
def test_gf2_inverse_classical_sim(m):
    bloq = GF2Inverse(m)
    GFM = GF(2**m)
    assert_consistent_classical_action(bloq, x=GFM.elements[1:])


@pytest.mark.parametrize('m', [*range(1, 12)])
def test_gf2_equivalent_bloq_counts(m):
    bloq = GF2Inverse(m)
    assert_equivalent_bloq_counts(bloq, generalizer=[ignore_split_join, ignore_alloc_free])
