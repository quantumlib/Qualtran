#  Copyright 2025 Google LLC
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

import itertools
import random

import pytest

from qualtran.rotation_synthesis.relative_norm import _clifford_t
from qualtran.rotation_synthesis.rings import _zw

_BASES = [_zw.Omega**i for i in range(4)]


@pytest.mark.parametrize("a", range(-10, 10))
@pytest.mark.parametrize("b", range(-10, 10))
def test_simple_cases(a, b):
    w = a + _zw.Omega * b
    target = (w * w.conjugate()).to_zsqrt2()[0]
    solver = _clifford_t.CliffordTRelativeNormSolver()
    got = solver.solve(target)
    assert got is not None
    assert (got * got.conjugate()).to_zsqrt2()[0] == target, f"{got=} {w=} {target}"


@pytest.mark.parametrize(
    "m", random.sample(tuple(itertools.product(range(-5, 5), repeat=4)), k=200)
)
def test_all_possiblities(m):
    w = _zw.ZW(m)
    target = (w * w.conjugate()).to_zsqrt2()[0]
    solver = _clifford_t.CliffordTRelativeNormSolver()
    got = solver.solve(target)
    assert got is not None
    assert (got * got.conjugate()).to_zsqrt2()[0] == target, f"{got=} {w=} {target} {m=}"
