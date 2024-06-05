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

from qualtran.symbolics import is_symbolic

from .data_types import (
    BoundedQUInt,
    check_dtypes_consistent,
    QAny,
    QAnyInt,
    QBit,
    QDType,
    QFxp,
    QInt,
    QIntOnesComp,
    QMontgomeryUInt,
    QUInt,
)


def test_qint():
    qint_8 = QInt(8)
    assert qint_8.num_qubits == 8
    assert str(qint_8) == 'QInt(8)'
    n = sympy.symbols('x')
    qint_8 = QInt(n)
    assert qint_8.num_qubits == n
    assert str(qint_8) == 'QInt(x)'
    assert is_symbolic(QInt(sympy.Symbol('x')))


def test_qint_ones():
    qint_8 = QIntOnesComp(8)
    assert str(qint_8) == 'QIntOnesComp(8)'
    assert qint_8.num_qubits == 8
    with pytest.raises(ValueError, match="num_qubits must be > 1."):
        QIntOnesComp(1)
    n = sympy.symbols('x')
    qint_8 = QIntOnesComp(n)
    assert qint_8.num_qubits == n
    assert is_symbolic(QIntOnesComp(sympy.Symbol('x')))


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


def test_bounded_quint():
    qint_3 = BoundedQUInt(2, 3)
    assert str(qint_3) == 'BoundedQUInt(2, 3)'

    assert qint_3.bitsize == 2
    assert qint_3.iteration_length == 3
    with pytest.raises(ValueError, match="BoundedQUInt iteration length.*"):
        BoundedQUInt(4, 76)
    n = sympy.symbols('x')
    l = sympy.symbols('l')
    qint_8 = BoundedQUInt(n, l)
    assert qint_8.num_qubits == n
    assert qint_8.iteration_length == l
    assert is_symbolic(BoundedQUInt(sympy.Symbol('x'), 2))
    assert is_symbolic(BoundedQUInt(2, sympy.Symbol('x')))
    assert is_symbolic(BoundedQUInt(*sympy.symbols('x y')))


def test_qfxp():
    qfp_16 = QFxp(16, 15)
    assert str(qfp_16) == 'QFxp(16, 15)'
    assert qfp_16.num_qubits == 16
    assert qfp_16.num_int == 1
    assert qfp_16.fxp_dtype_str == 'fxp-u16/15'
    qfp_16 = QFxp(16, 15, signed=True)
    assert str(qfp_16) == 'QFxp(16, 15, True)'
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
    assert is_symbolic(QFxp(*sympy.symbols('x y')))


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


@pytest.mark.parametrize('qdtype', [QBit(), QInt(4), QUInt(4), BoundedQUInt(3, 5)])
def test_domain_and_validation(qdtype: QDType):
    for v in qdtype.get_classical_domain():
        qdtype.assert_valid_classical_val(v)


@pytest.mark.parametrize('qdtype', [QBit(), QInt(4), QUInt(4), BoundedQUInt(3, 5)])
def test_domain_and_validation_arr(qdtype: QDType):
    arr = np.array(list(qdtype.get_classical_domain()))
    qdtype.assert_valid_classical_val_array(arr)


def test_validation_errs():
    with pytest.raises(ValueError):
        QBit().assert_valid_classical_val(-1)

    with pytest.raises(ValueError):
        QBit().assert_valid_classical_val('|0>')  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        QUInt(3).assert_valid_classical_val(8)

    with pytest.raises(ValueError):
        BoundedQUInt(3, 5).assert_valid_classical_val(-1)

    with pytest.raises(ValueError):
        BoundedQUInt(3, 5).assert_valid_classical_val(6)

    with pytest.raises(ValueError):
        QInt(4).assert_valid_classical_val(-9)

    with pytest.raises(ValueError):
        QUInt(3).assert_valid_classical_val(-1)

    with pytest.raises(ValueError):
        QUInt(3).assert_valid_classical_val(-1)


def test_validate_arrays():
    rs = np.random.RandomState(52)
    arr = rs.choice([0, 1], size=(23, 4))
    QBit().assert_valid_classical_val_array(arr)

    arr = rs.choice([-1, 1], size=(23, 4))
    with pytest.raises(ValueError):
        QBit().assert_valid_classical_val_array(arr)


@pytest.mark.parametrize(
    'qdtype', [QIntOnesComp(4), QFxp(4, 4), QInt(4), QUInt(4), BoundedQUInt(4, 5)]
)
def test_qany_consistency(qdtype):
    # All Types with correct bitsize are ok with QAny
    assert check_dtypes_consistent(qdtype, QAny(4))


@pytest.mark.parametrize('qdtype', [QUInt(4), BoundedQUInt(4, 5), QMontgomeryUInt(4)])
def test_type_errors_fxp_uint(qdtype):
    assert check_dtypes_consistent(qdtype, QFxp(4, 4))
    assert check_dtypes_consistent(qdtype, QFxp(4, 0))
    assert not check_dtypes_consistent(qdtype, QFxp(4, 2))
    assert not check_dtypes_consistent(qdtype, QFxp(4, 3, True))
    assert not check_dtypes_consistent(qdtype, QFxp(4, 0, True))


@pytest.mark.parametrize('qdtype', [QInt(4), QIntOnesComp(4)])
def test_type_errors_fxp_int(qdtype):
    assert not check_dtypes_consistent(qdtype, QFxp(4, 0))
    assert not check_dtypes_consistent(qdtype, QFxp(4, 4))


def test_type_errors_fxp():
    assert not check_dtypes_consistent(QFxp(4, 4), QFxp(4, 0))
    assert not check_dtypes_consistent(QFxp(4, 3, signed=True), QFxp(4, 0))
    assert not check_dtypes_consistent(QFxp(4, 3), QFxp(4, 0))


@pytest.mark.parametrize(
    'qdtype_a', [QUInt(4), BoundedQUInt(4, 5), QMontgomeryUInt(4), QInt(4), QIntOnesComp(4)]
)
@pytest.mark.parametrize(
    'qdtype_b', [QUInt(4), BoundedQUInt(4, 5), QMontgomeryUInt(4), QInt(4), QIntOnesComp(4)]
)
def test_type_errors_matrix(qdtype_a, qdtype_b):
    if qdtype_a == qdtype_b:
        assert check_dtypes_consistent(qdtype_a, qdtype_b)
    elif isinstance(qdtype_a, QAnyInt) and isinstance(qdtype_b, QAnyInt):
        assert check_dtypes_consistent(qdtype_a, qdtype_b)
    else:
        assert not check_dtypes_consistent(qdtype_a, qdtype_b)


def test_single_qubit_consistency():
    assert str(QBit()) == 'QBit()'
    assert check_dtypes_consistent(QBit(), QBit())
    assert check_dtypes_consistent(QBit(), QInt(1))
    assert check_dtypes_consistent(QInt(1), QBit())
    assert check_dtypes_consistent(QAny(1), QBit())
    assert check_dtypes_consistent(BoundedQUInt(1), QBit())
    assert check_dtypes_consistent(QFxp(1, 1), QBit())


def test_to_and_from_bits():
    # QInt
    qint4 = QInt(4)
    assert [*qint4.get_classical_domain()] == [*range(-8, 8)]
    for x in range(-8, 8):
        assert qint4.from_bits(qint4.to_bits(x)) == x
    assert list(qint4.to_bits(-2)) == [1, 1, 1, 0]
    assert list(QInt(4).to_bits(2)) == [0, 0, 1, 0]
    assert qint4.from_bits(qint4.to_bits(-2)) == -2
    assert qint4.from_bits(qint4.to_bits(2)) == 2
    with pytest.raises(ValueError):
        QInt(4).to_bits(10)

    # QUInt
    quint4 = QUInt(4)
    assert [*quint4.get_classical_domain()] == [*range(0, 16)]
    assert list(quint4.to_bits(10)) == [1, 0, 1, 0]
    assert quint4.from_bits(quint4.to_bits(10)) == 10
    for x in range(16):
        assert quint4.from_bits(quint4.to_bits(x)) == x
    with pytest.raises(ValueError):
        quint4.to_bits(16)

    with pytest.raises(ValueError):
        quint4.to_bits(-1)

    # BoundedQUInt
    bquint4 = BoundedQUInt(4, 12)
    assert [*bquint4.get_classical_domain()] == [*range(0, 12)]
    assert list(bquint4.to_bits(10)) == [1, 0, 1, 0]
    with pytest.raises(ValueError):
        BoundedQUInt(4, 12).to_bits(13)

    # QBit
    assert list(QBit().to_bits(0)) == [0]
    assert list(QBit().to_bits(1)) == [1]
    with pytest.raises(ValueError):
        QBit().to_bits(2)

    # QAny
    assert list(QAny(4).to_bits(10)) == [1, 0, 1, 0]

    # QIntOnesComp
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

    # QFxp: Negative numbers are stored as ones complement
    qfxp_4_3 = QFxp(4, 3, True)
    assert list(qfxp_4_3.to_bits(0.5)) == [0, 1, 0, 0]
    assert qfxp_4_3.from_bits(qfxp_4_3.to_bits(0.5)).get_val() == 0.5
    assert list(qfxp_4_3.to_bits(-0.5)) == [1, 1, 0, 0]
    assert qfxp_4_3.from_bits(qfxp_4_3.to_bits(-0.5)).get_val() == -0.5
    assert list(qfxp_4_3.to_bits(0.625)) == [0, 1, 0, 1]
    assert qfxp_4_3.from_bits(qfxp_4_3.to_bits(+0.625)).get_val() == +0.625
    assert qfxp_4_3.from_bits(qfxp_4_3.to_bits(-0.625)).get_val() == -0.625
    assert list(QFxp(4, 3, True).to_bits(-(1 - 0.625))) == [1, 1, 0, 1]
    assert qfxp_4_3.from_bits(qfxp_4_3.to_bits(0.375)).get_val() == 0.375
    assert qfxp_4_3.from_bits(qfxp_4_3.to_bits(-0.375)).get_val() == -0.375
    with pytest.raises(ValueError):
        _ = qfxp_4_3.to_bits(0.1)

    with pytest.raises(ValueError):
        _ = qfxp_4_3.to_bits(1.5)

    assert qfxp_4_3.from_bits(qfxp_4_3.to_bits(1 / 2 + 1 / 4 + 1 / 8)) == 1 / 2 + 1 / 4 + 1 / 8
    assert qfxp_4_3.from_bits(qfxp_4_3.to_bits(-1 / 2 - 1 / 4 - 1 / 8)) == -1 / 2 - 1 / 4 - 1 / 8
    with pytest.raises(ValueError):
        _ = qfxp_4_3.to_bits(1 / 2 + 1 / 4 + 1 / 8 + 1 / 16)

    for qfxp in [QFxp(4, 3, True), QFxp(3, 3, False), QFxp(7, 3, False), QFxp(7, 3, True)]:
        for x in qfxp.get_classical_domain():
            assert qfxp.from_bits(qfxp.to_bits(x)) == x

    assert list(QFxp(7, 3, True).to_bits(-4.375)) == [1] + [0, 1, 1] + [1, 0, 1]
    assert list(QFxp(7, 3, True).to_bits(+4.625)) == [0] + [1, 0, 0] + [1, 0, 1]
