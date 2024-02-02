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

from qualtran._infra.data_types import BoundedQInt, QFixedPoint, QInt, QIntOnesComp, QUnsignedInt


def test_qint():
    qint_8 = QInt(8)
    assert qint_8.num_qubits == 8


def test_qint_ones():
    qint_8 = QIntOnesComp(8)
    assert qint_8.num_qubits == 8


def test_quint():
    qint_8 = QUnsignedInt(8)
    assert qint_8.num_qubits == 8


def test_bounded_qint():
    qint_3 = BoundedQInt(2, 3)
    assert qint_3.bitsize == 2
    assert qint_3.iteration_length == 3
    with pytest.raises(ValueError):
        BoundedQInt(4, 76)


def test_qfixedpoint():
    qfp_16 = QFixedPoint(1, 15)
    assert qfp_16.num_qubits == 17
