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
import cirq
import numpy as np
import pytest

from qualtran.bloqs.qft.qft_text_book import QFTTextBook
from qualtran.testing import assert_valid_bloq_decomposition


@pytest.mark.parametrize('n', [2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize('without_reverse', [True, False])
def test_qft_school_book(n: int, without_reverse: bool):
    qft_bloq = QFTTextBook(n, not without_reverse)
    qft_cirq = cirq.QuantumFourierTransformGate(n, without_reverse=without_reverse)

    assert np.allclose(cirq.unitary(qft_bloq), cirq.unitary(qft_cirq))
    assert np.allclose(cirq.unitary(qft_bloq**-1), cirq.unitary(qft_cirq**-1))

    assert_valid_bloq_decomposition(qft_bloq)


@pytest.mark.parametrize('n', [10, 100, 500])
def test_qft_text_book_t_complexity(n: int):
    qft_bloq = QFTTextBook(n)
    qft_t_complexity = qft_bloq.t_complexity()
    assert qft_t_complexity.rotations == (n * (n - 1)) // 2
    assert qft_t_complexity.t == 0
