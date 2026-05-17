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
from typing import Any, Iterable, List, Sequence

import attrs
import galois
import numpy as np
import pytest
from numpy.typing import NDArray

from qualtran.dtype import BQUInt, QBit, QDType, QGF, QGFPoly, QInt, QUInt
from qualtran.symbolics import is_symbolic


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
        QGF(2, 8).assert_valid_classical_val(2**8)  # type: ignore[arg-type]

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


@attrs.frozen
class LegacyBQUInt(QDType):
    """For testing: this doesn't use a `BitEncoding`, so it will go via `_BitEncodingShim`"""

    bitsize: int
    iteration_length: int = attrs.field()

    @iteration_length.default
    def _default_iteration_length(self):
        return 2**self.bitsize

    def is_symbolic(self) -> bool:
        return is_symbolic(self.bitsize, self.iteration_length)

    @property
    def num_qubits(self):
        return self.bitsize

    def get_classical_domain(self) -> Iterable[Any]:
        if isinstance(self.iteration_length, int):
            return range(0, self.iteration_length)
        raise ValueError(f'Classical Domain not defined for expression: {self.iteration_length}')

    def assert_valid_classical_val(self, val: int, debug_str: str = 'val'):
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"{debug_str} should be an integer, not {val!r}")
        if val < 0:
            raise ValueError(f"Negative classical value encountered in {debug_str}")
        if val >= self.iteration_length:
            raise ValueError(f"Too-large classical value encountered in {debug_str}")

    def to_bits(self, x: int) -> List[int]:
        """Yields individual bits corresponding to binary representation of x"""
        self.assert_valid_classical_val(x, debug_str='val')
        return QUInt(self.bitsize).to_bits(x)

    def from_bits(self, bits: Sequence[int]) -> int:
        """Combine individual bits to form x"""
        return QUInt(self.bitsize).from_bits(bits)

    def assert_valid_classical_val_array(
        self, val_array: NDArray[np.integer], debug_str: str = 'val'
    ):
        if np.any(val_array < 0):
            raise ValueError(f"Negative classical values encountered in {debug_str}")
        if np.any(val_array >= self.iteration_length):
            raise ValueError(f"Too-large classical values encountered in {debug_str}")

    def __str__(self):
        return f'BQUInt({self.bitsize}, {self.iteration_length})'


def test_legacy_qcdtype():
    t = LegacyBQUInt(8, 230)
    assert str(t) == 'BQUInt(8, 230)'
    assert t._bit_encoding.to_bits(5) == QUInt(8).to_bits(5)
    t.assert_valid_classical_val(229)
    with pytest.raises(ValueError):
        t.assert_valid_classical_val(230)
