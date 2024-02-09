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

import sympy

from qualtran.bloqs.for_testing.interior_alloc import InteriorAlloc
from qualtran.bloqs.for_testing.with_decomposition import (
    TestIndependentParallelCombo,
    TestSerialCombo,
)
from qualtran.drawing import show_bloq
from qualtran.resource_counting._qubit_counting import _cbloq_max_width


def test_max_width_interior_alloc_symb():
    n = sympy.Symbol('n', positive=True)
    bloq = InteriorAlloc(n=n)
    show_bloq(bloq.decompose_bloq())

    binst_graph = bloq.decompose_bloq()._binst_graph
    max_width = _cbloq_max_width(binst_graph)
    assert max_width == 3 * n


def test_max_width_interior_alloc_nums():
    n = 10
    bloq = InteriorAlloc(n=n)
    show_bloq(bloq.decompose_bloq())

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
