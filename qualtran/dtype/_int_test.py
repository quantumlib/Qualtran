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

import pytest
import sympy

from qualtran.dtype import assert_to_and_from_bits_array_consistent, QInt
from qualtran.symbolics import is_symbolic


def test_qint():
    qint_8 = QInt(8)
    assert qint_8.num_qubits == 8
    assert qint_8.num_cbits == 0
    assert qint_8.num_bits == 8
    assert str(qint_8) == 'QInt(8)'
    n = sympy.symbols('x')
    qint_8 = QInt(n)
    assert qint_8.num_qubits == n
    assert str(qint_8) == 'QInt(x)'
    assert is_symbolic(QInt(sympy.Symbol('x')))


def test_qint_to_and_from_bits():
    qint4 = QInt(4)
    assert [*qint4.get_classical_domain()] == [*range(-8, 8)]
    for x in range(-8, 8):
        assert qint4.from_bits(qint4.to_bits(x)) == x
    assert list(qint4.to_bits(-2)) == [1, 1, 1, 0]
    assert list(QInt(4).to_bits(2)) == [0, 0, 1, 0]
    # MSB at lowest index -- big-endian
    assert qint4.from_bits([0, 0, 0, 1]) == 1
    assert qint4.from_bits([0, 0, 0, 1]) < qint4.from_bits([0, 1, 0, 0])
    assert qint4.from_bits(qint4.to_bits(-2)) == -2
    assert qint4.from_bits(qint4.to_bits(2)) == 2
    with pytest.raises(ValueError):
        QInt(4).to_bits(10)

    assert_to_and_from_bits_array_consistent(qint4, range(-8, 8))


def test_iter_bits_twos():
    assert QInt(4).to_bits(0) == [0, 0, 0, 0]
    assert QInt(4).to_bits(1) == [0, 0, 0, 1]
    assert QInt(4).to_bits(-2) == [1, 1, 1, 0]
    assert QInt(4).to_bits(-3) == [1, 1, 0, 1]
    with pytest.raises(ValueError):
        _ = QInt(2).to_bits(100)
