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

import numpy as np
import pytest
import sympy

from qualtran import BoundedQUInt, QBit, QDType, QFxp, QInt, QIntOnesComp, QUInt


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
    # works
    QUInt(1)
    n = sympy.symbols('x')
    qint_8 = QUInt(n)
    assert qint_8.num_qubits == n


def test_bounded_quint():
    qint_3 = BoundedQUInt(2, 3)
    assert qint_3.bitsize == 2
    assert qint_3.iteration_length == 3
    with pytest.raises(ValueError, match="BoundedQUInt iteration length.*"):
        BoundedQUInt(4, 76)
    n = sympy.symbols('x')
    l = sympy.symbols('l')
    qint_8 = BoundedQUInt(n, l)
    assert qint_8.num_qubits == n
    assert qint_8.iteration_length == l


def test_qfxp():
    qfp_16 = QFxp(16, 15)
    assert qfp_16.num_qubits == 16
    assert qfp_16.num_int == 1
    assert qfp_16.fxp_dtype_str == 'fxp-u16/15'
    qfp_16 = QFxp(16, 15, signed=True)
    assert qfp_16.num_qubits == 16
    assert qfp_16.num_int == 0
    assert qfp_16.fxp_dtype_str == 'fxp-s16/15'
    with pytest.raises(ValueError, match="num_qubits must be > 1."):
        QFxp(1, 1, signed=True)
    QFxp(1, 1, signed=False)
    with pytest.raises(ValueError, match="num_frac must be less than.*"):
        QFxp(4, 4, signed=True)
    with pytest.raises(ValueError, match="bitsize must be >= .*"):
        QFxp(4, 5)
    b = sympy.symbols('b')
    f = sympy.symbols('f')
    qfp = QFxp(b, f)
    assert qfp.num_qubits == b
    assert qfp.num_int == b - f
    qfp = QFxp(b, f, True)
    assert qfp.num_qubits == b
    assert qfp.num_int == b - f - 1


@pytest.mark.parametrize('qdtype', [QBit(), QInt(4), QUInt(4)])
def test_domain_and_validation(qdtype: QDType):
    for v in qdtype.get_classical_domain():
        qdtype.assert_valid_classical_val(v)


@pytest.mark.parametrize('qdtype', [QBit(), QInt(4), QUInt(4)])
def test_domain_and_validation_arr(qdtype: QDType):
    arr = np.array(list(qdtype.get_classical_domain()))
    qdtype.assert_valid_classical_val_array(arr)


def test_validation_errs():
    with pytest.raises(ValueError):
        QBit().assert_valid_classical_val(-1)

    with pytest.raises(ValueError):
        QBit().assert_valid_classical_val('|0>')

    with pytest.raises(ValueError):
        QUInt(3).assert_valid_classical_val(8)

    with pytest.raises(ValueError):
        QInt(4).assert_valid_classical_val(8)

    with pytest.raises(ValueError):
        QInt(4).assert_valid_classical_val(-9)

    with pytest.raises(ValueError):
        QUInt(3).assert_valid_classical_val(-1)


def test_validate_arrays():
    rs = np.random.RandomState(52)
    arr = rs.choice([0, 1], size=(23, 4))
    QBit().assert_valid_classical_val_array(arr)

    arr = rs.choice([-1, 1], size=(23, 4))
    with pytest.raises(ValueError):
        QBit().assert_valid_classical_val_array(arr)
