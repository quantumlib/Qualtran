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
from qualtran.bloqs.arithmetic.subtraction import (
    _sub_diff_size_regs,
    _sub_from_large,
    _sub_from_small,
    _sub_from_symb,
    _sub_large,
    _sub_small,
    _sub_symb,
    Subtract,
    SubtractFrom,
)
from qualtran.resource_counting.generalizers import ignore_split_join


def test_sub_symb(bloq_autotester):
    bloq_autotester(_sub_symb)


def test_sub_small(bloq_autotester):
    bloq_autotester(_sub_small)


def test_sub_large(bloq_autotester):
    bloq_autotester(_sub_large)


def test_sub_diff_size_regs(bloq_autotester):
    bloq_autotester(_sub_diff_size_regs)


def test_subtract_bloq_decomposition():
    gate = Subtract(QInt(3), QInt(5))
    qlt_testing.assert_valid_bloq_decomposition(gate)

    want = np.zeros((256, 256))
    for a_b in range(256):
        a, b = a_b >> 5, a_b & 31
        c = (a - b) % 32
        want[(a << 5) | c][a_b] = 1
    got = gate.tensor_contract()
    np.testing.assert_allclose(got, want)


def test_subtract_bloq_validation():
    assert Subtract(QUInt(3)) == Subtract(QUInt(3), QUInt(3))
    with pytest.raises(ValueError, match='bitsize must be less'):
        _ = Subtract(QInt(5), QInt(3))
    assert Subtract(QUInt(3)).dtype == QUInt(3)


def test_subtract_bloq_consistent_counts():
    qlt_testing.assert_equivalent_bloq_counts(
        Subtract(QInt(3), QInt(4)), generalizer=ignore_split_join
    )


def test_sub_from_symb(bloq_autotester):
    bloq_autotester(_sub_from_symb)


def test_sub_from_small(bloq_autotester):
    bloq_autotester(_sub_from_small)


def test_sub_from_large(bloq_autotester):
    bloq_autotester(_sub_from_large)


def test_subtract_from_bloq_decomposition():
    gate = SubtractFrom(QInt(4))
    qlt_testing.assert_valid_bloq_decomposition(gate)

    want = np.zeros((256, 256))
    for a_b in range(256):
        a, b = a_b >> 4, a_b & 15
        c = (b - a) % 16
        want[(a << 4) | c][a_b] = 1
    got = gate.tensor_contract()
    np.testing.assert_allclose(got, want)


def test_subtract_from_bloq_consistent_counts():
    qlt_testing.assert_equivalent_bloq_counts(SubtractFrom(QInt(3)), generalizer=ignore_split_join)
