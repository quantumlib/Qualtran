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

import pytest
import sympy

from qualtran.bloqs.factoring.mod_sub import MontgomeryModNeg, MontgomeryModSub
from qualtran.testing import assert_valid_bloq_decomposition


@pytest.mark.parametrize('bitsize,p', [(1, 1), (2, 3), (5, 8)])
def test_montgomery_mod_neg_decomp(bitsize, p):
    bloq = MontgomeryModNeg(bitsize=bitsize, p=p)
    assert_valid_bloq_decomposition(bloq)


@pytest.mark.parametrize('bitsize,p', [(1, 1), (2, 3), (5, 8)])
def test_montgomery_mod_sub_decomp(bitsize, p):
    bloq = MontgomeryModSub(bitsize=bitsize, p=p)
    assert_valid_bloq_decomposition(bloq)


@pytest.mark.parametrize('bitsize', [*range(1, 5), sympy.Symbol('n')])
def test_montgomery_sub_complexity(bitsize):
    tcomplexity = MontgomeryModSub(bitsize, sympy.Symbol('p')).t_complexity()
    assert tcomplexity.t == 24 * bitsize - 12  # 6n toffoli
    assert tcomplexity.rotations == 0


@pytest.mark.parametrize('bitsize', range(1, 5))
def test_montgomery_neg_complexity(bitsize):
    tcomplexity = MontgomeryModNeg(bitsize, sympy.Symbol('p')).t_complexity()
    assert tcomplexity.t == 8 * bitsize - 8  # 2n toffoli
    assert tcomplexity.rotations == 0


@pytest.mark.parametrize(
    ['prime', 'bitsize'],
    [(p, bitsize) for p in [11, 13, 31] for bitsize in range(1 + p.bit_length(), 8)],
)
def test_classical_action_montgomery_sub(bitsize, prime):
    b = MontgomeryModSub(bitsize, prime)
    cb = b.decompose_bloq()
    valid_range = range(prime)
    for x in valid_range:
        for y in valid_range:
            assert b.call_classically(x=x, y=y) == cb.call_classically(x=x, y=y)


@pytest.mark.parametrize(
    ['prime', 'bitsize'],
    [(p, bitsize) for p in [11, 13, 31] for bitsize in range(1 + p.bit_length(), 8)],
)
def test_classical_action_mod_neg(bitsize, prime):
    b = MontgomeryModNeg(bitsize, prime)
    cb = b.decompose_bloq()
    valid_range = range(prime)
    for x in valid_range:
        assert b.call_classically(x=x) == cb.call_classically(x=x)
