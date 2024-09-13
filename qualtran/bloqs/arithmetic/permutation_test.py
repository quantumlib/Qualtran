#  Copyright 2024 Google LLC
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
import sympy

import qualtran.testing as qlt_testing
from qualtran import QBit
from qualtran.bloqs.arithmetic.permutation import (
    _permutation,
    _permutation_cycle,
    _permutation_cycle_symb,
    _permutation_cycle_symb_N,
    _permutation_symb,
    _permutation_symb_with_cycles,
    _sparse_permutation,
    _sparse_permutation_with_symbolic_N,
    Permutation,
    PermutationCycle,
)
from qualtran.bloqs.basic_gates import CNOT, XGate
from qualtran.bloqs.bookkeeping import Allocate, Free
from qualtran.bloqs.mcmt import And
from qualtran.resource_counting.generalizers import generalize_cvs, ignore_split_join
from qualtran.symbolics import ceil, log2, slen


@pytest.mark.parametrize(
    "bloq_ex",
    [
        _permutation_cycle,
        _permutation,
        _sparse_permutation,
        _permutation_cycle_symb,
        _permutation_cycle_symb_N,
        _permutation_symb,
        _permutation_symb_with_cycles,
        _sparse_permutation_with_symbolic_N,
    ],
    ids=lambda bloq_ex: bloq_ex.name,
)
def test_examples(bloq_autotester, bloq_ex):
    bloq_autotester(bloq_ex)


@pytest.mark.parametrize(
    "bloq_ex",
    [
        _permutation_cycle,
        _permutation,
        _sparse_permutation,
        _permutation_cycle_symb_N,
        _permutation_symb_with_cycles,
        _sparse_permutation_with_symbolic_N,
    ],
    ids=lambda bloq_ex: bloq_ex.name,
)
def test_decomposition(bloq_ex):
    qlt_testing.assert_valid_bloq_decomposition(bloq_ex.make())


def test_permutation_cycle_unitary_and_call_graph():
    bloq = PermutationCycle(4, (0, 1, 2))

    np.testing.assert_allclose(
        bloq.tensor_contract(), np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    )

    cv = sympy.Symbol('cv')
    _, sigma = bloq.call_graph(
        generalizer=[ignore_split_join, generalize_cvs], keep=lambda b: isinstance(b, And)
    )
    assert sigma == {
        CNOT(): 8,
        And(cv1=cv, cv2=cv): 4,
        And(cv1=cv, cv2=cv).adjoint(): 4,
        Allocate(QBit()): 1,
        Free(QBit()): 1,
    }


def test_permutation_cycle_symbolic_call_graph():
    bloq = _permutation_cycle_symb()
    logN, L = ceil(log2(bloq.N)), slen(bloq.cycle)

    _, sigma = bloq.call_graph(keep=lambda b: isinstance(b, And))
    assert sigma == {
        And(): (L + 1) * (logN - 1),
        And().adjoint(): (L + 1) * (logN - 1),
        CNOT(): L * logN + L + 1,
    }


def test_permutation_unitary_and_call_graph():
    bloq = Permutation.from_dense_permutation([1, 2, 0, 4, 3, 5, 6])

    np.testing.assert_allclose(
        bloq.tensor_contract(),
        np.array(
            [
                [0, 0, 1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        ),
    )

    _, sigma = bloq.call_graph(generalizer=ignore_split_join, keep=lambda b: isinstance(b, And))
    assert sigma == {
        CNOT(): 17,
        And(): 56 // 4,
        And().adjoint(): 56 // 4,
        XGate(): 56,
        Allocate(QBit()): 2,
        Free(QBit()): 2,
    }


def test_sparse_permutation_classical_sim():
    N = 50
    perm_map = {0: 1, 1: 20, 2: 30, 3: 40}
    bloq = Permutation.from_partial_permutation_map(N, perm_map)
    assert bloq.bitsize == 6
    assert bloq.cycles == ((0, 1, 20), (2, 30), (3, 40))

    for i, x in perm_map.items():
        assert bloq.call_classically(x=i) == (x,)


def test_permutation_symbolic_call_graph():
    N = sympy.Symbol("N", positive=True, integer=True)
    logN = ceil(log2(N))
    bloq = _permutation_symb()

    _, sigma = bloq.call_graph(keep=lambda b: isinstance(b, And))
    assert sigma == {
        And().adjoint(): (N + 1) * (logN - 1),
        And(): (N + 1) * (logN - 1),
        CNOT(): N * logN + N + 1,
    }
