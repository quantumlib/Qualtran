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

from qualtran.dtype import QMontgomeryUInt
from qualtran.symbolics import is_symbolic


def test_qmontgomeryuint():
    qmontgomeryuint_8 = QMontgomeryUInt(8)
    assert str(qmontgomeryuint_8) == 'QMontgomeryUInt(8)'
    assert qmontgomeryuint_8.num_qubits == 8
    # works
    QMontgomeryUInt(1)
    n = sympy.symbols('x')
    qmontgomeryuint_8 = QMontgomeryUInt(n)
    assert qmontgomeryuint_8.num_qubits == n
    assert is_symbolic(QMontgomeryUInt(sympy.Symbol('x')))


@pytest.mark.parametrize('p', [13, 17, 29])
@pytest.mark.parametrize('val', [1, 5, 7, 9])
def test_qmontgomeryuint_operations(val, p):
    qmontgomeryuint_8 = QMontgomeryUInt(8, p)
    # Convert value to montgomery form and get the modular inverse.
    val_m = qmontgomeryuint_8.uint_to_montgomery(val)
    mod_inv = qmontgomeryuint_8.montgomery_inverse(val_m)

    # Calculate the product in montgomery form and convert back to normal form for assertion.
    assert (
        qmontgomeryuint_8.montgomery_to_uint(qmontgomeryuint_8.montgomery_product(val_m, mod_inv))
        == 1
    )


@pytest.mark.parametrize('p', [13, 17, 29])
@pytest.mark.parametrize('val', [1, 5, 7, 9])
def test_qmontgomeryuint_conversions(val, p):
    qmontgomeryuint_8 = QMontgomeryUInt(8, p)
    assert val == qmontgomeryuint_8.montgomery_to_uint(qmontgomeryuint_8.uint_to_montgomery(val))


@pytest.mark.parametrize('bitsize', range(1, 6))
def test_montgomery_bit_conversion(bitsize):
    dtype = QMontgomeryUInt(bitsize)
    for v in range(1 << bitsize):
        assert v == dtype.from_bits(dtype.to_bits(v))
