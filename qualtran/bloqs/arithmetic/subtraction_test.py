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

import itertools

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


@pytest.mark.parametrize(
    ['a_bits', 'b_bits'], [(a, b) for a in range(1, 6) for b in range(a, 6) if a + b <= 10]
)
def test_subtract_bloq_decomposition_unsigned(a_bits, b_bits):
    gate = Subtract(QUInt(a_bits), QUInt(b_bits))
    qlt_testing.assert_valid_bloq_decomposition(gate)

    tot = 1 << (a_bits + b_bits)
    want = np.zeros((tot, tot))
    max_b = 1 << b_bits
    for a_b in range(tot):
        a, b = a_b >> b_bits, a_b & (max_b - 1)
        c = (a - b) % max_b
        want[(a << b_bits) | c][a_b] = 1
    got = gate.tensor_contract()
    np.testing.assert_allclose(got, want)


def _to_signed_binary(x: int, bits: int):
    if x >= 0:
        return x
    return (~(-x) + 1) % (2 << bits)


@pytest.mark.parametrize(
    ['a_bits', 'b_bits'], [(a, b) for a in range(1, 6) for b in range(a, 6) if a + b <= 10]
)
def test_subtract_bloq_decomposition_signed(a_bits, b_bits):
    gate = Subtract(QInt(a_bits + 1), QInt(b_bits + 1))
    qlt_testing.assert_valid_bloq_decomposition(gate)

    tot = 1 << (a_bits + b_bits + 2)
    want = np.zeros((tot, tot))
    for a in range(-(1 << a_bits), (1 << a_bits)):
        for b in range(-(1 << b_bits), (1 << b_bits)):
            a_binary = _to_signed_binary(a, a_bits)
            b_binary = _to_signed_binary(b, b_bits)
            c_binary = _to_signed_binary(a - b, b_bits)
            want[(a_binary << b_bits << 1) | c_binary, (a_binary << b_bits << 1) | b_binary] = 1
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


@pytest.mark.parametrize('n_bits', range(1, 10))
def test_t_complexity(n_bits):
    complexity = Subtract(QUInt(n_bits)).t_complexity()
    assert complexity.t == 4 * (n_bits - 1)
    assert complexity.rotations == 0


@pytest.mark.parametrize('dtype', [QInt, QUInt])
def test_against_classical_values(dtype):
    subtract = Subtract(dtype(3), dtype(5))
    cbloq = subtract.decompose_bloq()
    if dtype is QInt:
        R1 = range(-4, 4)
        R2 = range(-16, 16)
    else:
        R1 = range(8)
        R2 = range(32)
    for (a, b) in itertools.product(R1, R2):
        ref = subtract.call_classically(a=a, b=b)
        comp = cbloq.call_classically(a=a, b=b)
        assert ref == comp


@pytest.mark.parametrize('bitsize', range(2, 5))
def test_classical_add_signed_overflow(bitsize):
    bloq = Subtract(QInt(bitsize))
    cbloq = bloq.decompose_bloq()
    mn = -(2 ** (bitsize - 1))
    assert bloq.call_classically(a=0, b=mn) == (0, mn)
    assert cbloq.call_classically(a=0, b=mn) == (0, mn)


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
