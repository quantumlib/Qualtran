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

from qualtran.dtype import QIntSignMag


def test_basic_properties():
    dtype = QIntSignMag(4)
    assert dtype.num_qubits == 4
    assert not dtype.is_symbolic()
    assert str(dtype) == 'QIntSignMag(4)'


def test_bitsize_validation():
    with pytest.raises(ValueError, match="bitsize must be >= 2"):
        QIntSignMag(1)


def test_to_bits_positive():
    dtype = QIntSignMag(4)
    assert dtype.to_bits(5) == [0, 1, 0, 1]
    assert dtype.to_bits(0) == [0, 0, 0, 0]
    assert dtype.to_bits(7) == [0, 1, 1, 1]


def test_to_bits_negative():
    dtype = QIntSignMag(4)
    assert dtype.to_bits(-5) == [1, 1, 0, 1]
    assert dtype.to_bits(-1) == [1, 0, 0, 1]
    assert dtype.to_bits(-7) == [1, 1, 1, 1]


def test_from_bits():
    dtype = QIntSignMag(4)
    assert dtype.from_bits([0, 1, 0, 1]) == 5
    assert dtype.from_bits([1, 1, 0, 1]) == -5
    assert dtype.from_bits([0, 0, 0, 0]) == 0
    assert dtype.from_bits([1, 0, 0, 0]) == 0  # -0 == 0


def test_roundtrip():
    # to_bits -> from_bits should be identity for all valid values.
    dtype = QIntSignMag(4)
    for val in dtype.get_classical_domain():
        bits = dtype.to_bits(val)
        recovered = dtype.from_bits(bits)
        assert recovered == val, f"Failed for {val}: bits={bits}, recovered={recovered}"


def test_classical_domain():
    dtype = QIntSignMag(4)
    domain = list(dtype.get_classical_domain())
    # 4-bit sign-magnitude: range is [-7, 7]
    assert domain == list(range(-7, 8))


def test_valid_classical_val():
    dtype = QIntSignMag(4)
    dtype.assert_valid_classical_val(0)
    dtype.assert_valid_classical_val(7)
    dtype.assert_valid_classical_val(-7)


def test_invalid_classical_val():
    dtype = QIntSignMag(4)
    with pytest.raises(ValueError, match="Too-large"):
        dtype.assert_valid_classical_val(8)
    with pytest.raises(ValueError, match="Too-small"):
        dtype.assert_valid_classical_val(-8)
