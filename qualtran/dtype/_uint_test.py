#  Copyright 2026 Google LLC
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
import sympy

from qualtran.dtype import assert_to_and_from_bits_array_consistent, QUInt
from qualtran.symbolics import is_symbolic


def test_quint():
    qint_8 = QUInt(8)
    assert str(qint_8) == 'QUInt(8)'

    assert qint_8.num_qubits == 8
    # works
    QUInt(1)
    n = sympy.symbols('x')
    qint_8 = QUInt(n)
    assert qint_8.num_qubits == n
    assert is_symbolic(QUInt(sympy.Symbol('x')))


def test_quint_to_and_from_bits():
    quint4 = QUInt(4)
    assert [*quint4.get_classical_domain()] == [*range(0, 16)]
    assert list(quint4.to_bits(10)) == [1, 0, 1, 0]
    assert quint4.from_bits(quint4.to_bits(10)) == 10
    # MSB at lowest index -- big-endian
    assert quint4.from_bits([0, 0, 0, 1]) == 1
    assert quint4.from_bits([0, 0, 0, 1]) < quint4.from_bits([1, 0, 0, 0])

    for x in range(16):
        assert quint4.from_bits(quint4.to_bits(x)) == x
    with pytest.raises(ValueError):
        quint4.to_bits(16)

    with pytest.raises(ValueError):
        quint4.to_bits(-1)

    assert_to_and_from_bits_array_consistent(quint4, range(0, 16))


def test_bits_to_int():
    cirq = pytest.importorskip('cirq')
    rs = np.random.RandomState(52)
    bitstrings = rs.choice([0, 1], size=(100, 23))

    nums = QUInt(23).from_bits_array(bitstrings)
    assert nums.shape == (100,)

    for num, bs in zip(nums, bitstrings):
        ref_num = cirq.big_endian_bits_to_int(bs.tolist())
        assert num == ref_num

    # check one input bitstring instead of array of input bitstrings.
    (num,) = QUInt(2).from_bits_array(np.array([1, 0]))
    assert num == 2


def test_int_to_bits():
    cirq = pytest.importorskip('cirq')
    rs = np.random.RandomState(52)
    nums = rs.randint(0, 2**23 - 1, size=(100,), dtype=np.uint64)
    bitstrings = QUInt(23).to_bits_array(nums)
    assert bitstrings.shape == (100, 23)

    for num, bs in zip(nums, bitstrings):
        ref_bs = cirq.big_endian_int_to_bits(int(num), bit_count=23)
        np.testing.assert_array_equal(ref_bs, bs)

    # check bounds
    with pytest.raises(AssertionError):
        QUInt(8).to_bits_array(np.array([4, -2]))


def test_iter_bits():
    assert QUInt(2).to_bits(0) == [0, 0]
    assert QUInt(2).to_bits(1) == [0, 1]
    assert QUInt(2).to_bits(2) == [1, 0]
    assert QUInt(2).to_bits(3) == [1, 1]
