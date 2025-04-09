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
import math
import random
from typing import Any, Sequence, Union

import galois
import numpy as np
import pytest
import sympy
from numpy.typing import NDArray

from qualtran import (
    BQUInt,
    CBit,
    check_dtypes_consistent,
    QAny,
    QBit,
    QDType,
    QFxp,
    QGF,
    QGFPoly,
    QInt,
    QIntOnesComp,
    QMontgomeryUInt,
    QUInt,
)
from qualtran._infra.data_types import _QAnyInt
from qualtran.symbolics import ceil, is_symbolic, log2


def test_bit():
    qbit = QBit()
    assert qbit.num_qubits == 1
    assert qbit.num_cbits == 0
    assert qbit.num_bits == 1
    assert str(qbit) == 'QBit()'

    cbit = CBit()
    assert cbit.num_cbits == 1
    assert cbit.num_qubits == 0
    assert cbit.num_bits == 1
    assert str(CBit()) == 'CBit()'


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
    qint_3 = BQUInt(2, 3)
    assert str(qint_3) == 'BQUInt(2, 3)'

    assert qint_3.bitsize == 2
    assert qint_3.iteration_length == 3
    with pytest.raises(ValueError, match="BQUInt iteration length.*"):
        BQUInt(4, 76)
    n = sympy.symbols('x')
    l = sympy.symbols('l')
    qint_8 = BQUInt(n, l)
    assert qint_8.num_qubits == n
    assert qint_8.iteration_length == l
    assert is_symbolic(BQUInt(sympy.Symbol('x'), 2))
    assert is_symbolic(BQUInt(2, sympy.Symbol('x')))
    assert is_symbolic(BQUInt(*sympy.symbols('x y')))


def test_qfxp():
    qfp_16 = QFxp(16, 15)
    assert str(qfp_16) == 'QFxp(16, 15)'
    assert qfp_16.num_qubits == 16
    assert qfp_16.num_int == 1
    assert qfp_16.fxp_dtype_template().dtype == 'fxp-u16/15'

    qfp_16 = QFxp(16, 15, signed=True)
    assert str(qfp_16) == 'QFxp(16, 15, True)'
    assert qfp_16.num_qubits == 16
    assert qfp_16.num_int == 1
    assert qfp_16.fxp_dtype_template().dtype == 'fxp-s16/15'

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
    assert qfp.num_int == b - f
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


def test_qgf():
    qgf_256 = QGF(characteristic=2, degree=8)
    assert str(qgf_256) == 'QGF(2**8)'
    assert qgf_256.num_qubits == 8
    p, m = sympy.symbols('p, m', integer=True, positive=True)
    qgf_pm = QGF(characteristic=p, degree=m)
    assert qgf_pm.num_qubits == ceil(log2(p**m))
    assert is_symbolic(qgf_pm)


def test_qgf_poly():
    qgf_poly_4_8 = QGFPoly(4, QGF(characteristic=2, degree=3))
    assert str(qgf_poly_4_8) == 'QGFPoly(4, QGF(2**3))'
    assert qgf_poly_4_8.num_qubits == 5 * 3
    n, p, m = sympy.symbols('n, p, m', integer=True, positive=True)
    qgf_poly_n_pm = QGFPoly(n, QGF(characteristic=p, degree=m))
    assert qgf_poly_n_pm.num_qubits == (n + 1) * ceil(log2(p**m))
    assert is_symbolic(qgf_poly_n_pm)


@pytest.mark.parametrize('qdtype', [QBit(), QInt(4), QUInt(4), BQUInt(3, 5)])
def test_domain_and_validation(qdtype: QDType):
    for v in qdtype.get_classical_domain():
        qdtype.assert_valid_classical_val(v)


@pytest.mark.parametrize(
    'qdtype',
    [
        QBit(),
        QInt(4),
        QUInt(4),
        BQUInt(3, 5),
        QGF(2, 8),
        QGFPoly(4, QGF(characteristic=2, degree=2)),
    ],
)
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
        BQUInt(3, 5).assert_valid_classical_val(-1)

    with pytest.raises(ValueError):
        BQUInt(3, 5).assert_valid_classical_val(6)

    with pytest.raises(ValueError):
        QInt(4).assert_valid_classical_val(-9)

    with pytest.raises(ValueError):
        QUInt(3).assert_valid_classical_val(-1)

    with pytest.raises(ValueError):
        QUInt(3).assert_valid_classical_val(-1)

    with pytest.raises(ValueError):
        QGF(2, 8).assert_valid_classical_val(2**8)

    with pytest.raises(ValueError):
        qgf = QGF(2, 3)
        poly = galois.Poly(qgf.gf_type([1, 2, 3, 4, 5, 6, 7]), field=qgf.gf_type)
        QGFPoly(4, qgf).assert_valid_classical_val(poly)


def test_validate_arrays():
    rs = np.random.RandomState(52)
    arr = rs.choice([0, 1], size=(23, 4))
    QBit().assert_valid_classical_val_array(arr)

    arr = rs.choice([-1, 1], size=(23, 4))
    with pytest.raises(ValueError):
        QBit().assert_valid_classical_val_array(arr)


@pytest.mark.parametrize('qdtype', [QIntOnesComp(4), QFxp(4, 4), QInt(4), QUInt(4), BQUInt(4, 5)])
def test_qany_consistency(qdtype):
    # All Types with correct bitsize are ok with QAny
    assert check_dtypes_consistent(qdtype, QAny(4))


@pytest.mark.parametrize('qdtype', [QUInt(4), BQUInt(4, 5), QMontgomeryUInt(4)])
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
    'qdtype_a', [QUInt(4), BQUInt(4, 5), QMontgomeryUInt(4), QInt(4), QIntOnesComp(4)]
)
@pytest.mark.parametrize(
    'qdtype_b', [QUInt(4), BQUInt(4, 5), QMontgomeryUInt(4), QInt(4), QIntOnesComp(4)]
)
def test_type_errors_matrix(qdtype_a, qdtype_b):
    if qdtype_a == qdtype_b:
        assert check_dtypes_consistent(qdtype_a, qdtype_b)
    elif isinstance(qdtype_a, _QAnyInt) and isinstance(qdtype_b, _QAnyInt):
        assert check_dtypes_consistent(qdtype_a, qdtype_b)
    else:
        assert not check_dtypes_consistent(qdtype_a, qdtype_b)


def test_single_qubit_consistency():
    assert str(QBit()) == 'QBit()'
    assert check_dtypes_consistent(QBit(), QBit())
    assert check_dtypes_consistent(QBit(), QInt(1))
    assert check_dtypes_consistent(QInt(1), QBit())
    assert check_dtypes_consistent(QAny(1), QBit())
    assert check_dtypes_consistent(BQUInt(1), QBit())
    assert check_dtypes_consistent(QFxp(1, 1), QBit())
    assert check_dtypes_consistent(QGF(characteristic=2, degree=1), QBit())


def assert_to_and_from_bits_array_consistent(qdtype: QDType, values: Union[Sequence[Any], NDArray]):
    values = np.asanyarray(values)
    bits_array = qdtype.to_bits_array(values)

    # individual values
    for val, bits in zip(values.reshape(-1), bits_array.reshape(-1, qdtype.num_qubits)):
        assert np.all(bits == qdtype.to_bits(val))

    # round trip
    values_roundtrip = qdtype.from_bits_array(bits_array)
    assert np.all(values_roundtrip == values)


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


def test_qgf_to_and_from_bits():
    from galois import GF

    qgf_256 = QGF(2, 8)
    gf256 = GF(2**8)
    assert [*qgf_256.get_classical_domain()] == [*range(256)]
    a, b = qgf_256.to_bits(gf256(21)), qgf_256.to_bits(gf256(22))
    c = qgf_256.from_bits(list(np.bitwise_xor(a, b)))
    assert c == gf256(21) + gf256(22)
    for x in gf256.elements:
        assert x == gf256.Vector(qgf_256.to_bits(x))

    with pytest.raises(ValueError):
        qgf_256.to_bits(21)
    assert_to_and_from_bits_array_consistent(qgf_256, gf256([*range(256)]))


def test_qgf_poly_to_and_from_bits():
    qgf_4 = QGF(2, 2)
    qgf_poly_2_4 = QGFPoly(2, qgf_4)
    assert_to_and_from_bits_array_consistent(qgf_poly_2_4, [*qgf_poly_2_4.get_classical_domain()])


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


def test_bounded_quint_to_and_from_bits():
    bquint4 = BQUInt(4, 12)
    assert [*bquint4.get_classical_domain()] == [*range(0, 12)]
    assert list(bquint4.to_bits(10)) == [1, 0, 1, 0]
    with pytest.raises(ValueError):
        BQUInt(4, 12).to_bits(13)

    assert_to_and_from_bits_array_consistent(bquint4, range(0, 12))


def test_qbit_to_and_from_bits():
    assert list(QBit().to_bits(0)) == [0]
    assert list(QBit().to_bits(1)) == [1]
    with pytest.raises(ValueError):
        QBit().to_bits(2)

    assert_to_and_from_bits_array_consistent(QBit(), [0, 1])


def test_qany_to_and_from_bits():
    assert list(QAny(4).to_bits(10)) == [1, 0, 1, 0]

    assert_to_and_from_bits_array_consistent(QAny(4), range(16))


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


def test_qfxp_to_and_from_bits():
    assert_to_and_from_bits_array_consistent(
        QFxp(4, 3, False), [QFxp(4, 3, False).to_fixed_width_int(x) for x in [1 / 2, 1 / 4, 3 / 8]]
    )
    assert_to_and_from_bits_array_consistent(
        QFxp(4, 3, True),
        [
            QFxp(4, 3, True).to_fixed_width_int(x)
            for x in [1 / 2, -1 / 2, 1 / 4, -1 / 4, -3 / 8, 3 / 8]
        ],
    )


def test_qfxp_to_fixed_width_int():
    assert QFxp(6, 4).to_fixed_width_int(1.5) == 24 == 1.5 * 2**4
    assert QFxp(6, 4, signed=True).to_fixed_width_int(1.5) == 24 == 1.5 * 2**4
    assert QFxp(6, 4, signed=True).to_fixed_width_int(-1.5) == -24 == -1.5 * 2**4


def test_qfxp_from_fixed_width_int():
    qfxp = QFxp(6, 4)
    for x_int in qfxp.get_classical_domain():
        x_float = qfxp.float_from_fixed_width_int(x_int)
        x_int_roundtrip = qfxp.to_fixed_width_int(x_float)
        assert x_int == x_int_roundtrip

    for float_val in [1.5, 1.25]:
        assert qfxp.float_from_fixed_width_int(qfxp.to_fixed_width_int(float_val)) == float_val


def test_qfxp_to_and_from_bits_using_fxp():
    # QFxp: Negative numbers are stored as twos complement
    qfxp_4_3 = QFxp(4, 3, True)
    assert list(qfxp_4_3._fxp_to_bits(0.5)) == [0, 1, 0, 0]
    assert qfxp_4_3._from_bits_to_fxp(qfxp_4_3._fxp_to_bits(0.5)).get_val() == 0.5
    assert list(qfxp_4_3._fxp_to_bits(-0.5)) == [1, 1, 0, 0]
    assert qfxp_4_3._from_bits_to_fxp(qfxp_4_3._fxp_to_bits(-0.5)).get_val() == -0.5
    assert list(qfxp_4_3._fxp_to_bits(0.625)) == [0, 1, 0, 1]
    assert qfxp_4_3._from_bits_to_fxp(qfxp_4_3._fxp_to_bits(+0.625)).get_val() == +0.625
    assert qfxp_4_3._from_bits_to_fxp(qfxp_4_3._fxp_to_bits(-0.625)).get_val() == -0.625
    assert list(qfxp_4_3._fxp_to_bits(-(1 - 0.625))) == [1, 1, 0, 1]
    assert qfxp_4_3._from_bits_to_fxp(qfxp_4_3._fxp_to_bits(0.375)).get_val() == 0.375
    assert qfxp_4_3._from_bits_to_fxp(qfxp_4_3._fxp_to_bits(-0.375)).get_val() == -0.375
    with pytest.raises(ValueError):
        _ = qfxp_4_3._fxp_to_bits(0.1)
    assert list(qfxp_4_3._fxp_to_bits(0.7, require_exact=False)) == [0, 1, 0, 1]
    assert list(qfxp_4_3._fxp_to_bits(0.7, require_exact=False, complement=False)) == [0, 1, 0, 1]
    assert list(qfxp_4_3._fxp_to_bits(-0.7, require_exact=False)) == [1, 0, 1, 1]
    assert list(qfxp_4_3._fxp_to_bits(-0.7, require_exact=False, complement=False)) == [1, 1, 0, 1]

    with pytest.raises(ValueError):
        _ = qfxp_4_3._fxp_to_bits(1.5)

    assert (
        qfxp_4_3._from_bits_to_fxp(qfxp_4_3._fxp_to_bits(1 / 2 + 1 / 4 + 1 / 8))
        == 1 / 2 + 1 / 4 + 1 / 8
    )
    assert (
        qfxp_4_3._from_bits_to_fxp(qfxp_4_3._fxp_to_bits(-1 / 2 - 1 / 4 - 1 / 8))
        == -1 / 2 - 1 / 4 - 1 / 8
    )
    with pytest.raises(ValueError):
        _ = qfxp_4_3._fxp_to_bits(1 / 2 + 1 / 4 + 1 / 8 + 1 / 16)

    for qfxp in [QFxp(4, 3, True), QFxp(3, 3, False), QFxp(7, 3, False), QFxp(7, 3, True)]:
        for x in qfxp._get_classical_domain_fxp():
            assert qfxp._from_bits_to_fxp(qfxp._fxp_to_bits(x)) == x

    assert list(QFxp(7, 3, True)._fxp_to_bits(-4.375)) == [1] + [0, 1, 1] + [1, 0, 1]
    assert list(QFxp(7, 3, True)._fxp_to_bits(+4.625)) == [0] + [1, 0, 0] + [1, 0, 1]


def test_iter_bits():
    assert QUInt(2).to_bits(0) == [0, 0]
    assert QUInt(2).to_bits(1) == [0, 1]
    assert QUInt(2).to_bits(2) == [1, 0]
    assert QUInt(2).to_bits(3) == [1, 1]


def test_iter_bits_twos():
    assert QInt(4).to_bits(0) == [0, 0, 0, 0]
    assert QInt(4).to_bits(1) == [0, 0, 0, 1]
    assert QInt(4).to_bits(-2) == [1, 1, 1, 0]
    assert QInt(4).to_bits(-3) == [1, 1, 0, 1]
    with pytest.raises(ValueError):
        _ = QInt(2).to_bits(100)


random.seed(1234)


@pytest.mark.parametrize('val', [random.uniform(-1, 1) for _ in range(10)])
@pytest.mark.parametrize('width', [*range(2, 20, 2)])
@pytest.mark.parametrize('signed', [True, False])
def test_fixed_point(val, width, signed):
    if (val < 0) and not signed:
        with pytest.raises(ValueError):
            _ = QFxp(width + int(signed), width, signed=signed)._fxp_to_bits(
                val, require_exact=False, complement=False
            )
    else:
        bits = QFxp(width + int(signed), width, signed=signed)._fxp_to_bits(
            val, require_exact=False, complement=False
        )
        if signed:
            sign, bits = bits[0], bits[1:]
            assert sign == (1 if val < 0 else 0)
        val = abs(val)
        approx_val = math.fsum([b * (1 / 2 ** (1 + i)) for i, b in enumerate(bits)])
        assert math.isclose(val, approx_val, abs_tol=1 / 2**width), (
            f'{val}:{approx_val}:{width}',
            bits,
        )
        with pytest.raises(ValueError):
            _ = QFxp(width, width).to_fixed_width_int(-val)
        bits_from_int = QUInt(width).to_bits(QFxp(width, width).to_fixed_width_int(val))
        assert bits == bits_from_int


@pytest.mark.parametrize('bitsize', range(1, 6))
def test_montgomery_bit_conversion(bitsize):
    dtype = QMontgomeryUInt(bitsize)
    for v in range(1 << bitsize):
        assert v == dtype.from_bits(dtype.to_bits(v))


def test_qgf_with_default_poly_is_compatible():
    qgf_one = QGF(2, 4)

    qgf_two = QGF(2, 4, irreducible_poly=qgf_one.gf_type.irreducible_poly)

    assert qgf_one == qgf_two
