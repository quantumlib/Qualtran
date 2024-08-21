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
from qualtran.linalg.permutation import (
    decompose_permutation_into_cycles,
    decompose_permutation_map_into_cycles,
)


def test_decompose_permutation_into_cycles():
    assert list(decompose_permutation_into_cycles([0, 1, 2])) == []
    assert list(decompose_permutation_into_cycles([1, 2, 0])) == [(0, 1, 2)]
    assert sorted(decompose_permutation_into_cycles([1, 0, 2, 4, 5, 3])) == [(0, 1), (3, 4, 5)]


def test_decompose_sparse_prefix_permutation_into_cycles():
    assert list(decompose_permutation_map_into_cycles({0: 1, 1: 20})) == [(0, 1, 20)]
    assert sorted(decompose_permutation_map_into_cycles({0: 30, 1: 50})) == [(0, 30), (1, 50)]
    assert list(decompose_permutation_map_into_cycles({0: 0})) == []
    assert list(decompose_permutation_map_into_cycles({0: 1, 1: 0})) == [(0, 1)]
