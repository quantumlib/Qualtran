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

import math

import pytest
from attrs import frozen

from qualtran.surface_code.fifteen_to_one import FifteenToOne
from qualtran.surface_code.magic_count import MagicCount


@frozen
class TestCase:
    d_X: int
    d_Z: int
    d_m: int
    phys_err: float

    p_out: float
    footprint: int
    cycles: float


PAPER_RESULTS = [
    TestCase(d_X=7, d_Z=3, d_m=3, phys_err=1e-4, p_out=4.4e-8, footprint=810, cycles=18.1),
    TestCase(d_X=9, d_Z=3, d_m=3, phys_err=1e-4, p_out=9.3e-10, footprint=1150, cycles=18.1),
    TestCase(d_X=11, d_Z=5, d_m=5, phys_err=1e-4, p_out=1.9e-11, footprint=2070, cycles=30),
    TestCase(d_X=17, d_Z=7, d_m=7, phys_err=1e-3, p_out=4.5e-8, footprint=4620, cycles=42.6),
]


@pytest.mark.parametrize("test", PAPER_RESULTS)
def test_compare_with_paper(test: TestCase):
    factory = FifteenToOne(test.d_X, test.d_Z, test.d_m)
    assert f'{factory.distillation_error(MagicCount(n_t=1), test.phys_err):.1e}' == str(test.p_out)
    assert round(factory.footprint(), -1) == test.footprint  # rounding to the 10s digit.
    assert factory.n_cycles(MagicCount(n_t=1), test.phys_err) == math.ceil(test.cycles + 1e-9)


def test_validation():
    for bad_args in (1, 1, -1), (1, -1, 1), (-1, 1, 1), (5, 1, 1):
        with pytest.raises(AssertionError):
            _ = FifteenToOne(*bad_args)
