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

import qualtran.testing as qlt_testing
from qualtran.bloqs.arithmetic.sorting import _bitonic_sort, _cmp_symb, BitonicSort, Comparator
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.symbolics import ceil, log2


def test_cmp_symb(bloq_autotester):
    bloq_autotester(_cmp_symb)


def test_bitonic_sort(bloq_autotester):
    bloq_autotester(_bitonic_sort)


def test_comparator_manual():
    bloq = Comparator(2**4)
    assert bloq.t_complexity().t == 88 - 4 - 7 * 4


def test_comparator_symbolic_t_complexity():
    bloq = _cmp_symb.make()
    bitsize = ceil(log2(bloq.L - 1))
    assert bloq.t_complexity() == TComplexity(t=15 * bitsize + 4, clifford=56 * bitsize + 24)


def test_bitonic_sort_manual():
    bitsize = 4
    k = 8

    bloq = BitonicSort(2**bitsize, k)
    assert bloq.num_comparisons == 24

    _ = bloq.t_complexity()


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('sorting')
