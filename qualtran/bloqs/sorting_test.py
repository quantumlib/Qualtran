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

from qualtran.bloqs.sorting import BitonicSort, Comparator
from qualtran.testing import execute_notebook


def _make_comparator():
    from qualtran.bloqs.sorting import Comparator

    return Comparator(bitsize=4)


def _make_bitonic_sort():
    from qualtran.bloqs.sorting import BitonicSort

    return BitonicSort(bitsize=8, k=8)


def test_comparator():
    bloq = Comparator(4)
    assert bloq.t_complexity().t == 88
    with pytest.raises(NotImplementedError):
        bloq.decompose_bloq()


def test_bitonic_sort():
    bitsize = 4
    k = 8
    bloq = BitonicSort(bitsize, k)
    assert bloq.t_complexity().t == 8 * 9 * 88
    with pytest.raises(NotImplementedError):
        bloq.decompose_bloq()


def test_notebook():
    execute_notebook('sorting')
