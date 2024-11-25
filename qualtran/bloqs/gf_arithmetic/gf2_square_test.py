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

from qualtran.bloqs.gf_arithmetic.gf2_square import _gf2_square_symbolic, _gf16_square, GF2Square
from qualtran.resource_counting import get_cost_value, QECGatesCost
from qualtran.symbolics import ceil, log2
from qualtran.testing import assert_consistent_classical_action


def test_gf16_square(bloq_autotester):
    bloq_autotester(_gf16_square)


def test_gf2_square_symbolic(bloq_autotester):
    bloq_autotester(_gf2_square_symbolic)


def test_gf2_square_classical_sim_quick():
    m = 2
    bloq = GF2Square(m)
    GFM = GF(2**m)
    assert_consistent_classical_action(bloq, x=GFM.elements)


def test_gf2_square_resource():
    bloq = _gf2_square_symbolic.make()
    m = bloq.bitsize
    assert get_cost_value(bloq, QECGatesCost()).total_t_count() == 0
    assert sympy.simplify(get_cost_value(bloq, QECGatesCost()).clifford - ceil(m**2 / log2(m))) == 0


@pytest.mark.slow
@pytest.mark.parametrize('m', [3, 4, 5])
def test_gf2_square_classical_sim(m):
    bloq = GF2Square(m)
    GFM = GF(2**m)
    assert_consistent_classical_action(bloq, x=GFM.elements)
