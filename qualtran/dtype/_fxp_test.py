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
import math

import numpy as np
import pytest
import sympy

from qualtran.dtype import assert_to_and_from_bits_array_consistent, QFxp, QUInt
from qualtran.dtype._fxp import _Fxp
from qualtran.symbolics import is_symbolic

RS = np.random.RandomState(52)


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
    qfxp_4_3 = _Fxp(4, 3, True)
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
        for x in qfxp._bit_encoding._get_domain_fxp():
            assert qfxp._bit_encoding._from_bits_to_fxp(qfxp._bit_encoding._fxp_to_bits(x)) == x

    assert list(_Fxp(7, 3, True)._fxp_to_bits(-4.375)) == [1] + [0, 1, 1] + [1, 0, 1]
    assert list(_Fxp(7, 3, True)._fxp_to_bits(+4.625)) == [0] + [1, 0, 0] + [1, 0, 1]


@pytest.mark.parametrize('val', [RS.uniform(-1, 1) for _ in range(10)])
@pytest.mark.parametrize('width', [*range(2, 20, 2)])
@pytest.mark.parametrize('signed', [True, False])
def test_fixed_point(val, width, signed):
    if (val < 0) and not signed:
        with pytest.raises(ValueError):
            _ = _Fxp(width + int(signed), width, signed=signed)._fxp_to_bits(
                val, require_exact=False, complement=False
            )
    else:
        bits = _Fxp(width + int(signed), width, signed=signed)._fxp_to_bits(
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
