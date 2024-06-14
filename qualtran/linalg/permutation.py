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

import numpy as np

CycleT: TypeAlias = tuple[int, ...]


def decompose_permutation_into_cycles(permutation: Sequence[int]) -> Iterator[CycleT]:
    """Generate all non-trivial (more than one element) cycles of a permutation of [0, ..., N - 1]"""
    import networkx as nx

    G = nx.DiGraph(enumerate(permutation))
    for cycle in nx.simple_cycles(G):
        if len(cycle) >= 2:
            yield tuple(cycle)


def decompose_sparse_prefix_permutation_into_cycles(
    permutation_prefix: Sequence[int], N: int
) -> Iterator[CycleT]:
    r"""Given a prefix of a permutation, extend it to a valid permutation and return non-trivial cycles.

    For some $d \le N$, we are given the prefix of a permutation on $N$, i.e.
    $\{ x_i | 0 \le i < d \}$. It is guaranteed that each input $x_i$ is unique, and
    therefore it can be extended to a valid permuation on $N$. This procedure generates
    a sequence of cycles such that the number of swaps required to implement this prefix
    is minimized.

    Args:
        permutation_prefix: a length $d$ prefix of a permutation on $N$.
        N: the total size of the generated permutation.
    """
    d = len(permutation_prefix)
    if d > N:
        raise ValueError(f"Permutation limit {N=} is smaller than the prefix {d=}")

    seen = np.full(d, False)

    for i in range(d):
        if seen[i]:
            continue

        # compute the cycle starting at `i`
        cycle = []
        while i < d and not seen[i]:
            seen[i] = True
            cycle.append(i)
            i = permutation_prefix[i]

        if i >= d:
            # cycle ends outside the prefix, so add this element to the cycle
            cycle.append(i)

        if len(cycle) >= 2:
            yield tuple(cycle)
