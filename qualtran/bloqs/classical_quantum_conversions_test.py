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

import qualtran.testing as qlt_testing
from qualtran.bloqs.classical_quantum_conversions import (
    ClassicalToQBit,
    ClassicalToQInt,
    ClassicalToQIntOnesComp,
    ClassicalToQUInt,
)


@pytest.mark.parametrize('k', [(0), (1)])
def test_classical_to_qbit_decomp(k):
    bloq = ClassicalToQBit(k=k)
    qlt_testing.assert_valid_bloq_decomposition(bloq)


@pytest.mark.parametrize('k,result', [(0, 0), (1, 1)])
def test_classical_to_qbit_classical_sim(k, result):
    bloq = ClassicalToQBit(k=k)
    cbloq = bloq.decompose_bloq()
    bloq_classical = bloq.call_classically(x=0)
    cbloq_classical = cbloq.call_classically(x=0)

    assert len(bloq_classical) == len(cbloq_classical)
    for i in range(len(bloq_classical)):
        np.testing.assert_array_equal(bloq_classical[i], cbloq_classical[i])

    assert bloq_classical[-1] == result


@pytest.mark.parametrize('bitsize,k', [(3, 2), (5, 8), (6, 30)])
def test_classical_to_qint_decomp(bitsize, k):
    bloq = ClassicalToQInt(bitsize=bitsize, k=k)
    qlt_testing.assert_valid_bloq_decomposition(bloq)


@pytest.mark.parametrize('bitsize,k,result', [(3, 2, 2), (5, 8, 8), (6, 30, 30)])
def test_classical_to_qint_classical_sim(bitsize, k, result):
    bloq = ClassicalToQInt(bitsize=bitsize, k=k)
    cbloq = bloq.decompose_bloq()
    bloq_classical = bloq.call_classically(x=0)
    cbloq_classical = cbloq.call_classically(x=0)

    assert len(bloq_classical) == len(cbloq_classical)
    for i in range(len(bloq_classical)):
        np.testing.assert_array_equal(bloq_classical[i], cbloq_classical[i])

    assert bloq_classical[-1] == result


@pytest.mark.parametrize('bitsize,k', [(3, 2), (5, 8), (6, 30)])
def test_classical_to_quint_decomp(bitsize, k):
    bloq = ClassicalToQUInt(bitsize=bitsize, k=k)
    qlt_testing.assert_valid_bloq_decomposition(bloq)


@pytest.mark.parametrize('bitsize,k,result', [(3, 2, 2), (5, 8, 8), (6, 30, 30)])
def test_classical_to_quint_classical_sim(bitsize, k, result):
    bloq = ClassicalToQUInt(bitsize=bitsize, k=k)
    cbloq = bloq.decompose_bloq()
    bloq_classical = bloq.call_classically(x=0)
    cbloq_classical = cbloq.call_classically(x=0)

    assert len(bloq_classical) == len(cbloq_classical)
    for i in range(len(bloq_classical)):
        np.testing.assert_array_equal(bloq_classical[i], cbloq_classical[i])

    assert bloq_classical[-1] == result


@pytest.mark.parametrize('bitsize,k', [(3, 2), (5, 8), (6, 30)])
def test_classical_to_qint_ones_comp_decomp(bitsize, k):
    bloq = ClassicalToQIntOnesComp(bitsize=bitsize, k=k)
    qlt_testing.assert_valid_bloq_decomposition(bloq)


@pytest.mark.parametrize('bitsize,k,result', [(3, 2, 2), (5, 8, 8), (6, 30, 30)])
def test_classical_to_qint_ones_comp_classical_sim(bitsize, k, result):
    bloq = ClassicalToQIntOnesComp(bitsize=bitsize, k=k)
    cbloq = bloq.decompose_bloq()
    bloq_classical = bloq.call_classically(x=0)
    cbloq_classical = cbloq.call_classically(x=0)

    assert len(bloq_classical) == len(cbloq_classical)
    for i in range(len(bloq_classical)):
        np.testing.assert_array_equal(bloq_classical[i], cbloq_classical[i])

    assert bloq_classical[-1] == result
