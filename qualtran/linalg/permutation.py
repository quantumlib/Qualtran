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
from typing import Iterator, Sequence, TypeAlias

CycleT: TypeAlias = tuple[int, ...]


def decompose_permutation_into_cycles(permutation: Sequence[int]) -> Iterator[CycleT]:
    """Generate all non-trivial (more than one element) cycles of a permutation of [0, ..., N - 1]"""
    import networkx as nx

    G = nx.DiGraph(enumerate(permutation))
    for cycle in nx.simple_cycles(G):
        if len(cycle) >= 2:
            yield tuple(cycle)


def decompose_permutation_map_into_cycles(permutation_map: dict[int, int]) -> Iterator[CycleT]:
    r"""Given a (partial) permutation map, return non-trivial cycles requiring minimum swaps.

    We are given a partial permutation on $N$ as a python dictionary. This procedure generates a
    sequence of cycles such that the number of swaps required to implement this partial mapping
    is minimized.

    >>> list(decompose_permutation_map_into_cycles({0: 1, 1: 5, 5: 0, 2: 6, 3: 3}, N=10))
    [(0, 1, 5), (2, 6)]

    Args:
        permutation_map: a (partial) map defining the permutation.
    """
    seen = set()

    for i in permutation_map:
        if i in seen:
            continue

        # compute the cycle starting at `i`
        cycle = []
        while i not in seen:
            seen.add(i)
            cycle.append(i)
            if i not in permutation_map:
                break
            i = permutation_map[i]

        if len(cycle) >= 2:
            yield tuple(cycle)
