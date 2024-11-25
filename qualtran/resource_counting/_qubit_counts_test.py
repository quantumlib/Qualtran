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

import pytest
import sympy

import qualtran.testing as qlt_testing
from qualtran import QAny
from qualtran.bloqs.basic_gates import Swap, TwoBitSwap
from qualtran.bloqs.bookkeeping import Allocate, Free
from qualtran.bloqs.for_testing.interior_alloc import InteriorAlloc
from qualtran.bloqs.for_testing.with_decomposition import (
    TestIndependentParallelCombo,
    TestSerialCombo,
)
from qualtran.drawing import show_bloq
from qualtran.resource_counting import get_cost_cache, get_cost_value, QubitCount
from qualtran.resource_counting._qubit_counts import _cbloq_max_width
from qualtran.resource_counting.generalizers import ignore_split_join


def test_max_width_interior_alloc_symb():
    n = sympy.Symbol('n', positive=True)
    bloq = InteriorAlloc(n=n)

    binst_graph = bloq.decompose_bloq()._binst_graph
    max_width = _cbloq_max_width(binst_graph)
    assert max_width == 3 * n


def test_max_width_interior_alloc_nums():
    n = 10
    bloq = InteriorAlloc(n=n)

    binst_graph = bloq.decompose_bloq()._binst_graph
    max_width = _cbloq_max_width(binst_graph)
    assert max_width == 30


def test_max_width_disconnected_components():
    bloq = TestIndependentParallelCombo()
    binst_graph = bloq.decompose_bloq()._binst_graph
    max_width = _cbloq_max_width(binst_graph)
    assert max_width == 1


def test_max_width_simple():
    show_bloq(TestSerialCombo().decompose_bloq())
    max_width = _cbloq_max_width(TestSerialCombo().decompose_bloq()._binst_graph)
    assert max_width == 1


def test_qubit_count_cost():
    bloq = InteriorAlloc(n=10)
    qubit_counts = get_cost_cache(bloq, QubitCount(), generalizer=ignore_split_join)
    assert qubit_counts == {
        InteriorAlloc(n=10): 30,
        Allocate(QAny(10)): 10,
        Free(QAny(10)): 10,
        Swap(10): 20,
        TwoBitSwap(): 2,
    }


def test_on_cbloq():
    n = sympy.Symbol('n', positive=True, integer=True)
    bloq = InteriorAlloc(n=n)
    cbloq = bloq.decompose_bloq()
    n_qubits = get_cost_value(cbloq, QubitCount())
    assert n_qubits == 3 * n


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook("qubit_counts")
