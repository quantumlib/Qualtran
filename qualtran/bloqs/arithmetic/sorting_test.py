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


def test_cmp_symb(bloq_autotester):
    bloq_autotester(_cmp_symb)


def test_bitonic_sort(bloq_autotester):
    bloq_autotester(_bitonic_sort)


def test_comparator_manual():
    bloq = Comparator(4)
    assert bloq.t_complexity().t == 88 - 4
    with pytest.raises(NotImplementedError):
        bloq.decompose_bloq()


def test_bitonic_sort_manual():
    bitsize = 4
    k = 8
    bloq = BitonicSort(bitsize, k)
    assert bloq.t_complexity().t == 8 * 9 * (88 - 4)
    with pytest.raises(NotImplementedError):
        bloq.decompose_bloq()


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('sorting')
