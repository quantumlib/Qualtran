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

from enum import Enum
from typing import Any, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from ._any import QAny
from ._base import QCDType, QDType
from ._buint import BQUInt
from ._fxp import QFxp
from ._int import QInt
from ._montgomery_uint import QMontgomeryUInt
from ._uint import QUInt
from .gf import QGF

_QAnyInt = (QInt, QUInt, BQUInt, QMontgomeryUInt)
_QAnyUInt = (QUInt, BQUInt, QMontgomeryUInt, QGF)


def assert_to_and_from_bits_array_consistent(qdtype: QDType, values: Union[Sequence[Any], NDArray]):
    values = np.asanyarray(values)
    bits_array = qdtype.to_bits_array(values)

    # individual values
    for val, bits in zip(values.reshape(-1), bits_array.reshape(-1, qdtype.num_qubits)):
        assert np.all(bits == qdtype.to_bits(val))

    # round trip
    values_roundtrip = qdtype.from_bits_array(bits_array)
    assert np.all(values_roundtrip == values)


class QDTypeCheckingSeverity(Enum):
    """The level of type checking to enforce"""

    LOOSE = 0
    """Allow most type conversions between QAnyInt, QFxp and QAny."""

    ANY = 1
    """Disallow numeric type conversions but allow QAny and single bit conversion."""

    STRICT = 2
    """Strictly enforce type checking between registers. Only single bit conversions are allowed."""


def _check_uint_fxp_consistent(
    a: Union['QUInt', 'BQUInt', 'QMontgomeryUInt', 'QGF'], b: 'QFxp'
) -> bool:
    """A uint / qfxp is consistent with a whole or totally fractional unsigned QFxp."""
    if b.signed:
        return False
    return a.num_qubits == b.num_qubits and (b.num_frac == 0 or b.num_int == 0)


def check_dtypes_consistent(
    dtype_a: QCDType,
    dtype_b: QCDType,
    type_checking_severity: QDTypeCheckingSeverity = QDTypeCheckingSeverity.LOOSE,
) -> bool:
    """Check if two types are consistent given our current definition on consistent types.

    Args:
        dtype_a: The dtype to check against the reference.
        dtype_b: The reference dtype.
        type_checking_severity: Severity of type checking to perform.

    Returns:
        True if the types are consistent.
    """
    same_dtypes = dtype_a == dtype_b
    same_n_qubits = dtype_a.num_qubits == dtype_b.num_qubits
    if same_dtypes:
        # Same types are always ok.
        return True
    elif dtype_a.num_qubits == 1 and same_n_qubits:
        # Single qubit types are ok.
        return True
    if type_checking_severity == QDTypeCheckingSeverity.STRICT:
        return False
    if isinstance(dtype_a, QAny) or isinstance(dtype_b, QAny):
        # QAny -> any dtype and any dtype -> QAny
        return same_n_qubits
    if type_checking_severity == QDTypeCheckingSeverity.ANY:
        return False
    if isinstance(dtype_a, _QAnyInt) and isinstance(dtype_b, _QAnyInt):
        # A subset of the integers should be freely interchangeable.
        return same_n_qubits
    elif isinstance(dtype_a, _QAnyUInt) and isinstance(dtype_b, QFxp):
        # unsigned Fxp which is wholly an integer or < 1 part is a uint.
        return _check_uint_fxp_consistent(dtype_a, dtype_b)
    elif isinstance(dtype_b, _QAnyUInt) and isinstance(dtype_a, QFxp):
        # unsigned Fxp which is wholy an integer or < 1 part is a uint.
        return _check_uint_fxp_consistent(dtype_b, dtype_a)
    else:
        return False
