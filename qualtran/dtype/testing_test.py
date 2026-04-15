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

from qualtran.dtype import (
    BQUInt,
    check_dtypes_consistent,
    QAny,
    QBit,
    QFxp,
    QGF,
    QInt,
    QIntOnesComp,
    QMontgomeryUInt,
    QUInt,
)
from qualtran.dtype.testing import _QAnyInt


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
