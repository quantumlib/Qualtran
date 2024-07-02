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

import heapq
from collections import Counter
from typing import Dict, Iterator, List, TYPE_CHECKING

import attrs
import networkx as nx

if TYPE_CHECKING:
    from qualtran import BloqInstance

_INFINITY: int = int(1e18)


def _priority(node: 'BloqInstance') -> int:
    from qualtran._infra.gate_with_registers import total_bits
    from qualtran._infra.quantum_graph import DanglingT
    from qualtran.bloqs.bookkeeping import Allocate, Free

    if isinstance(node, DanglingT):
        return 0

    if node.bloq_is(Allocate):
        return _INFINITY

    if node.bloq_is(Free):
        return -_INFINITY

    signature = node.bloq.signature
    return total_bits(signature.rights()) - total_bits(signature.lefts())


@attrs.frozen(order=True)
class _PrioritizedItem:
    """Helper dataclass to insert items in a heap as part of `greedy_topological_sort`."""

    item: 'BloqInstance' = attrs.field(eq=_priority, order=_priority)
    priority: int


def greedy_topological_sort(binst_graph: nx.DiGraph) -> Iterator['BloqInstance']:
    """Stable greedy topological sorting for the bloq instance graph.

    Topological sorting for the Bloq Instances graph which maintains a priority queue
    instead of a queue. Priority for each bloq is a tuple of the form
    (_priority(bloq), insertion_index); where each term corresponds to

    ### Priority of a bloq
     - 0: For Left / Right Dangling bloqs.
     - +Infinity / -Infinity: For `Allocate` / `Free` bloqs.
     - total_bits(right registers) - total_bits(left registers): For all other bloqs.

    ### Insertion Index
    `insertion_index` is a unique ID used to breaks ties between bloqs that have the
    same priority and follows from the order of nodes inserted in the networkx Graph.

    The stability condition guarantees that two networkx graphs constructed with
    identical ordering of Graph.nodes and Graph.edges will have the same topological
    sorting.

    Args:
        binst_graph: A networkx DiGraph with `BloqInstances` as nodes. Usually obtained
        from `cbloq._binst_graph` where `cbloq` is a `CompositeBloq`.

    Yields:
        Nodes from the input graph returned in a greedy topological sorted order with the
        goal to minimize qubit allocations and deallocations by pushing allocations to the
        right and de-allocations to the left.
    """
    heap: List[_PrioritizedItem] = []
    idx: int = 0
    in_degree: Dict[BloqInstance, int] = Counter()

    for x in binst_graph.nodes():
        in_degree[x] = binst_graph.in_degree(x)
        if not in_degree[x]:
            heapq.heappush(heap, _PrioritizedItem(x, idx))
            idx = idx + 1

    while heap:
        x = heapq.heappop(heap).item
        yield x
        for y in binst_graph.neighbors(x):
            in_degree[y] -= 1
            if not in_degree[y]:
                heapq.heappush(heap, _PrioritizedItem(y, idx))
                idx = idx + 1
