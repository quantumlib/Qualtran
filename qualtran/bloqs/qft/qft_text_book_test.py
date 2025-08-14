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
import sympy

import qualtran.testing as qlt_testing
from qualtran.bloqs.qft.qft_text_book import _qft_text_book, _symbolic_qft, QFTTextBook
from qualtran.resource_counting import get_cost_value, QECGatesCost
from qualtran.resource_counting.generalizers import ignore_split_join


@pytest.mark.parametrize('without_reverse', [True, False])
def test_qft_text_book_quick(without_reverse: bool):
    n = 3
    qft_bloq = QFTTextBook(n, not without_reverse)
    qft_cirq = cirq.QuantumFourierTransformGate(n, without_reverse=without_reverse)

    assert np.allclose(qft_bloq.tensor_contract(), cirq.unitary(qft_cirq))
    assert np.allclose(qft_bloq.adjoint().tensor_contract(), cirq.unitary(qft_cirq**-1))

    qlt_testing.assert_valid_bloq_decomposition(qft_bloq)


@pytest.mark.slow
@pytest.mark.parametrize('n', [2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize('without_reverse', [True, False])
def test_qft_text_book(n: int, without_reverse: bool):
    qft_bloq = QFTTextBook(n, not without_reverse)
    qft_cirq = cirq.QuantumFourierTransformGate(n, without_reverse=without_reverse)

    assert np.allclose(qft_bloq.tensor_contract(), cirq.unitary(qft_cirq))
    assert np.allclose(qft_bloq.adjoint().tensor_contract(), cirq.unitary(qft_cirq**-1))

    qlt_testing.assert_valid_bloq_decomposition(qft_bloq)


@pytest.mark.parametrize('n', [10, 123])
def test_qft_text_book_t_complexity(n: int):
    qft_bloq = QFTTextBook(n)
    qlt_testing.assert_equivalent_bloq_counts(qft_bloq, generalizer=[ignore_split_join])
    gate_counts = get_cost_value(qft_bloq, QECGatesCost())
    # special angle ZPow gets turned into clifford or T
    rots = ((n - 3) * (n - 2)) // 2
    if n >= 41:
        # TODO(https://github.com/quantumlib/Qualtran/issues/1474)
        pytest.xfail("Small angle rotations")
    assert gate_counts.t == n - 2
    assert gate_counts.toffoli == 0
    assert gate_counts.rotation == rots
    assert gate_counts.and_bloq == (n * (n - 1)) // 2


def test_qft_text_book_t_complexity_symbolic():
    n = sympy.symbols('n')
    qft_bloq = QFTTextBook(bitsize=n)
    gate_counts = get_cost_value(qft_bloq, QECGatesCost())
    assert gate_counts.rotation == (n - 1) * (n // 2)
    assert gate_counts.and_bloq == (n - 1) * (n // 2)


def test_qft_text_book_auto(bloq_autotester):
    bloq_autotester(_qft_text_book)


def test_symbolic_qft_auto(bloq_autotester):
    bloq_autotester(_symbolic_qft)
