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

from qualtran.dtype import assert_to_and_from_bits_array_consistent, CBit, QBit


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


def test_qbit_to_and_from_bits():
    assert list(QBit().to_bits(0)) == [0]
    assert list(QBit().to_bits(1)) == [1]
    with pytest.raises(ValueError):
        QBit().to_bits(2)

    assert_to_and_from_bits_array_consistent(QBit(), [0, 1])
