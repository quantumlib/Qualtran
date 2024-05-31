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

import numpy as np
import pytest

import qualtran.testing as qlt_testing
from qualtran import QInt, QUInt
from qualtran.bloqs.arithmetic import Subtract
from qualtran.resource_counting.generalizers import ignore_split_join


def test_subtract_bloq_decomposition():
    gate = Subtract(QInt(3), QInt(5))
    qlt_testing.assert_valid_bloq_decomposition(gate)

    want = np.zeros((256, 256))
    for a_b in range(256):
        a, b = a_b >> 5, a_b & 31
        c = (a - b) % 32
        want[(a << 5) | c][a_b] = 1
    got = gate.tensor_contract()
    np.testing.assert_equal(got, want)


def test_subtract_bloq_validation():
    assert Subtract(QUInt(3)) == Subtract(QUInt(3), QUInt(3))
    with pytest.raises(ValueError, match='bitsize must be less'):
        _ = Subtract(QInt(5), QInt(3))
    assert Subtract(QUInt(3)).dtype == QUInt(3)


def test_subtract_bloq_consitant_counts():
    qlt_testing.assert_equivalent_bloq_counts(
        Subtract(QInt(3), QInt(4)), generalizer=ignore_split_join
    )
