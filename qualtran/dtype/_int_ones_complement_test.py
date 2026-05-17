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

from qualtran.dtype import assert_to_and_from_bits_array_consistent, QIntOnesComp
from qualtran.symbolics import is_symbolic


def test_qint_ones():
    qint_8 = QIntOnesComp(8)
    assert str(qint_8) == 'QIntOnesComp(8)'
    assert qint_8.num_qubits == 8
    with pytest.raises(ValueError, match="bitsize must be > 1."):
        QIntOnesComp(1)
    n = sympy.symbols('x')
    qint_8 = QIntOnesComp(n)
    assert qint_8.num_qubits == n
    assert is_symbolic(QIntOnesComp(sympy.Symbol('x')))


def test_qintonescomp_to_and_from_bits():
    qintones4 = QIntOnesComp(4)
    assert list(qintones4.to_bits(-2)) == [1, 1, 0, 1]
    assert list(qintones4.to_bits(2)) == [0, 0, 1, 0]
    assert [*qintones4.get_classical_domain()] == [*range(-7, 8)]
    for x in range(-7, 8):
        assert qintones4.from_bits(qintones4.to_bits(x)) == x
    with pytest.raises(ValueError):
        qintones4.to_bits(8)
    with pytest.raises(ValueError):
        qintones4.to_bits(-8)

    assert_to_and_from_bits_array_consistent(qintones4, range(-7, 8))
