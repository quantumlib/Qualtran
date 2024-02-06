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

from qualtran._infra.data_types import BoundedQInt, QFixedPoint, QInt, QIntOnesComp, QUInt


def test_qint():
    qint_8 = QInt(8)
    assert qint_8.num_qubits == 8
    with pytest.raises(ValueError, match="num_qubits must be > 1."):
        QInt(1)
    n = sympy.symbols('x')
    qint_8 = QInt(n)
    assert qint_8.num_qubits == n


def test_qint_ones():
    qint_8 = QIntOnesComp(8)
    assert qint_8.num_qubits == 8
    with pytest.raises(ValueError, match="num_qubits must be > 1."):
        QIntOnesComp(1)
    n = sympy.symbols('x')
    qint_8 = QIntOnesComp(n)
    assert qint_8.num_qubits == n


def test_quint():
    qint_8 = QUInt(8)
    assert qint_8.num_qubits == 8
    with pytest.raises(ValueError, match="num_qubits must be > 1."):
        QUInt(1)
    n = sympy.symbols('x')
    qint_8 = QUInt(n)
    assert qint_8.num_qubits == n


def test_bounded_qint():
    qint_3 = BoundedQInt(2, 3)
    assert qint_3.bitsize == 2
    assert qint_3.iteration_length == 3
    with pytest.raises(ValueError, match="BoundedQInt iteration length.*"):
        BoundedQInt(4, 76)
    with pytest.raises(ValueError, match="num_qubits must be > 1."):
        BoundedQInt(1, 10)
    n = sympy.symbols('x')
    l = sympy.symbols('l')
    qint_8 = BoundedQInt(n, l)
    assert qint_8.num_qubits == n
    assert qint_8.iteration_length == l


def test_qfixedpoint():
    qfp_16 = QFixedPoint(1, 15)
    assert qfp_16.num_qubits == 17
    with pytest.raises(ValueError, match="num_qubits must be > 1."):
        QFixedPoint(0, 0)
    i = sympy.symbols('i')
    f = sympy.symbols('f')
    qint_8 = QFixedPoint(i, f)
    assert qint_8.num_qubits == i + f + sympy.symbols('1')
