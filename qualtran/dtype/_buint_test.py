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

from qualtran.dtype import assert_to_and_from_bits_array_consistent, BQUInt
from qualtran.symbolics import is_symbolic


def test_bounded_quint():
    qint_3 = BQUInt(2, 3)
    assert str(qint_3) == 'BQUInt(2, 3)'

    assert qint_3.bitsize == 2
    assert qint_3.iteration_length == 3
    with pytest.raises(ValueError, match="iteration length is too large.*"):
        BQUInt(4, 76)
    n = sympy.symbols('x')
    l = sympy.symbols('l')
    qint_8 = BQUInt(n, l)
    assert qint_8.num_qubits == n
    assert qint_8.iteration_length == l
    assert is_symbolic(BQUInt(sympy.Symbol('x'), 2))
    assert is_symbolic(BQUInt(2, sympy.Symbol('x')))
    assert is_symbolic(BQUInt(*sympy.symbols('x y')))


def test_bounded_quint_to_and_from_bits():
    bquint4 = BQUInt(4, 12)
    assert [*bquint4.get_classical_domain()] == [*range(0, 12)]
    assert list(bquint4.to_bits(10)) == [1, 0, 1, 0]
    with pytest.raises(ValueError):
        BQUInt(4, 12).to_bits(13)

    assert_to_and_from_bits_array_consistent(bquint4, range(0, 12))
