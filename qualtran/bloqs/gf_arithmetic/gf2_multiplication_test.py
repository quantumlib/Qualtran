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
from galois import GF

from qualtran.bloqs.gf_arithmetic.gf2_multiplication import (
    _gf2_multiplication_symbolic,
    _gf16_multiplication,
    GF2Multiplication,
)
from qualtran.testing import assert_consistent_classical_action


def test_gf16_multiplication(bloq_autotester):
    bloq_autotester(_gf16_multiplication)


def test_gf2_multiplication_symbolic(bloq_autotester):
    bloq_autotester(_gf2_multiplication_symbolic)


def test_gf2_multiplication_classical_sim_quick():
    m = 2
    bloq = GF2Multiplication(m)
    GFM = GF(2**m)
    assert_consistent_classical_action(bloq, x=GFM.elements, y=GFM.elements)


@pytest.mark.slow
@pytest.mark.parametrize('m', [3, 4, 5])
def test_gf2_multiplication_classical_sim(m):
    bloq = GF2Multiplication(m)
    GFM = GF(2**m)
    assert_consistent_classical_action(bloq, x=GFM.elements, y=GFM.elements)
