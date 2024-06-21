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

from itertools import permutations

import numpy as np
import pytest

import qualtran.testing as qlt_testing
from qualtran.bloqs.arithmetic.sorting import (
    _bitonic_merge,
    _bitonic_sort,
    _bitonic_sort_symb,
    _comparator,
    _comparator_symb,
    _parallel_compare,
    BitonicMerge,
    BitonicSort,
    Comparator,
    ParallelComparators,
)
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.testing import assert_valid_bloq_decomposition


def test_comparator_examples(bloq_autotester):
    bloq_autotester(_comparator)
    bloq_autotester(_comparator_symb)


def test_comparator_manual():
    bloq = Comparator(4)
    assert bloq.t_complexity().t == 88 - 4 - 7 * 4


def test_comparator_symbolic_t_complexity():
    bloq = _comparator_symb.make()
    assert bloq.t_complexity() == TComplexity(
        t=15 * bloq.bitsize + 4, clifford=56 * bloq.bitsize + 24
    )


@pytest.mark.parametrize("L", [5, 8, 12])
def test_comparator_classical_sim(L: int):
    bloq = Comparator(L)
    for a in range(L):
        for b in range(L):
            res_a, res_b, anc = bloq.call_classically(a=a, b=b)
            assert res_a <= res_b
            assert res_a == min(a, b)
            assert res_b == max(a, b)
            assert anc == (a > b)


def test_parallel_compare_example(bloq_autotester):
    bloq_autotester(_parallel_compare)


@pytest.mark.parametrize("k", [*range(1, 10)])
@pytest.mark.parametrize("offset", [*range(1, 10)])
def test_parallel_compare_decompose(k: int, offset: int):
    bitsize = 3

    bloq = ParallelComparators(k=k, offset=offset, bitsize=bitsize)
    assert_valid_bloq_decomposition(bloq)


def test_bitonic_merge_example(bloq_autotester):
    bloq_autotester(_bitonic_merge)


@pytest.mark.parametrize("k", [2, 4, 8, 16])
def test_bitonic_merge_classical_sim_on_random_lists(k: int):
    rng = np.random.default_rng(1024)

    bitsize = 3

    for _ in range(5):
        xs, ys = rng.integers(0, 2**bitsize, size=(2, k))
        xs = np.sort(xs)
        ys = np.sort(ys)
        result, _ = BitonicMerge(k, bitsize).call_classically(xs=xs, ys=ys)
        assert np.all(result == sorted([*xs, *ys]))


def test_bitonic_sort_examples(bloq_autotester):
    bloq_autotester(_bitonic_sort)
    bloq_autotester(_bitonic_sort_symb)


def test_bitonic_sort_manual():
    bitsize = 4
    k = 8

    bloq = BitonicSort(k, bitsize)
    assert bloq.num_comparisons == 24

    _ = bloq.t_complexity()


def test_bitonic_sort_classical_sim():
    bitsize = 3
    xs = np.array([4, 2, 7, 1])
    bloq = BitonicSort(len(xs), bitsize)
    sorted_xs, _ = bloq.call_classically(xs=xs)
    assert np.all(sorted_xs == sorted(xs))


@pytest.mark.parametrize("k", [1, 2, 4, 8, 16])
def test_bitonic_sort_classical_sim_on_random_lists(k: int):
    rng = np.random.default_rng(1024)

    bitsize = 3
    bloq = BitonicSort(k, bitsize)

    for _ in range(5):
        xs = rng.integers(0, 2**bitsize, size=k)
        sorted_xs, _ = bloq.call_classically(xs=xs)
        assert np.all(sorted_xs == sorted(xs))


def test_bitonic_sort_classical_sim_on_all_permutations():
    k = 4
    bloq = BitonicSort(k, k)
    for xs in permutations(range(k)):
        sorted_xs, _ = bloq.call_classically(xs=np.array(xs))
        assert np.all(sorted_xs == sorted(xs))


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('sorting')
