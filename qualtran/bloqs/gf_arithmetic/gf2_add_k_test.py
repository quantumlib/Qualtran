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

from qualtran.bloqs.gf_arithmetic.gf2_add_k import _gf2_add_k_symbolic, _gf16_add_k, GF2AddK
from qualtran.testing import assert_consistent_classical_action


def test_gf16_add_k(bloq_autotester):
    bloq_autotester(_gf16_add_k)


def test_gf2_add_k_symbolic(bloq_autotester):
    bloq_autotester(_gf2_add_k_symbolic)


def test_gf2_add_k_classical_sim_quick():
    m = 2
    GFM = GF(2**m)
    for k in GFM.elements:
        bloq = GF2AddK(m, int(k))
        assert_consistent_classical_action(bloq, x=GFM.elements)


@pytest.mark.slow
@pytest.mark.parametrize('m', [3, 4, 5])
def test_gf2_add_k_classical_sim(m):
    GFM = GF(2**m)
    for k in GFM.elements:
        bloq = GF2AddK(m, int(k))
        assert_consistent_classical_action(bloq, x=GFM.elements)
