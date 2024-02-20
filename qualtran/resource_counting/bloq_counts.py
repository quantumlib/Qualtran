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

"""Functionality for the `Bloq.call_graph()` protocol."""

from collections import defaultdict
from enum import Enum
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import networkx as nx
import sympy

from qualtran import Bloq, CompositeBloq, DecomposeNotImplementedError, DecomposeTypeError

from ._cost_key import BloqCount, CostKey
from ._cost_val import AddCostVal, CostVal, CostValT

BloqCountT = Tuple[Bloq, Union[int, sympy.Expr]]
GeneralizerT = Callable[[Bloq], Optional[Bloq]]
CostKV = Tuple[CostKey[CostValT], CostValT]


def big_O(expr) -> sympy.Order:
    """Helper to deal with CS-style big-O notation that takes the infinite limit by default."""
    if isinstance(expr, (int, float)):
        return sympy.Order(expr)
    return sympy.Order(expr, *[(var, sympy.oo) for var in expr.free_symbols])


class SympySymbolAllocator:
    """A class that allocates unique sympy symbols for integrating out bloq attributes.

    When counting, we group bloqs that only differ in attributes that do not affect
    resource costs. In practice, we do this by replacing bloqs with a version where
    the offending attributes have been set to an arbitrary (but unique) symbol allocated
    by this class. We refer to this process as "generalizing".
    """

    def __init__(self):
        self._idxs: Dict[str, int] = defaultdict(lambda: 0)

    def new_symbol(self, prefix: str) -> sympy.Symbol:
        """Return a unique symbol beginning with _prefix."""
        s = sympy.Symbol(f'_{prefix}{self._idxs[prefix]}')
        self._idxs[prefix] += 1
        return s


def build_cbloq_call_graph(cbloq: CompositeBloq) -> Set[BloqCountT]:
    """Count all the subbloqs in a composite bloq.

    This is the function underpinning `CompositeBloq.build_call_graph`.

    Args:
        cbloq: The composite bloq.
    """
    counts: Dict[Bloq, int] = defaultdict(lambda: 0)
    for binst in cbloq.bloq_instances:
        counts[binst.bloq] += 1

    return {(bloq, n) for bloq, n in counts.items()}


def _generalize_callees(
    raw_callee_counts: Set[BloqCountT], generalizer: GeneralizerT
) -> List[BloqCountT]:
    """Apply `generalizer` to the results of `bloq.build_call_graph`.

    This calls `generalizer` on each of the callees returned from that function,
    and filters out cases where `generalizer` returns `None`.
    """
    callee_counts: List[BloqCountT] = []
    for callee, n in raw_callee_counts:
        callee = generalizer(callee)
        if callee is None:
            # Signifies that this callee should be ignored.
            continue
        callee_counts.append((callee, n))
    return callee_counts


class _CallGraphNodeType(Enum):
    ALREADY_VISITED = 1
    LEAF = 2
    LEAF_FROM_GENERALIZATION = 3
    INTERNAL = 4


def _add_costs(g: nx.DiGraph, bloq: Bloq, costs: Iterable[CostKV]) -> None:
    data = g.nodes[bloq]
    if 'costs' in data:
        data['costs'].extend(costs)
    else:
        data['costs'] = list(costs)


def _build_call_graph(
    bloq: Bloq,
    generalizer: GeneralizerT,
    ssa: SympySymbolAllocator,
    keep: Callable[[Bloq], bool],
    max_depth: Optional[int],
    g: nx.DiGraph,
    depth: int,
) -> _CallGraphNodeType:
    """Recursively build the call graph.

    Arguments are the same as `get_bloq_call_graph`, except `g` is the graph we're building
    (i.e. it is mutated by this function) and `depth` is the current recursion depth.
    """
    if bloq in g:
        # We already visited this node.
        return _CallGraphNodeType.ALREADY_VISITED

    # Make sure this node is present in the graph and annotate
    # static costs.
    g.add_node(bloq)
    _add_costs(g, bloq, costs=bloq.my_static_costs())

    # Base case 1: This node is requested by the user to be a leaf node via the `keep` parameter.
    if keep(bloq):
        return _CallGraphNodeType.LEAF

    # Base case 2: Max depth exceeded
    if max_depth is not None and depth >= max_depth:
        return _CallGraphNodeType.LEAF

    # Prep for recursion: get the callees and modify them according to `generalizer`.
    try:
        callee_counts = _generalize_callees(bloq.build_call_graph(ssa), generalizer)
    except (DecomposeNotImplementedError, DecomposeTypeError):
        # Base case 3: Decomposition (or `bloq_counts`) is not implemented. This is left as a
        #              leaf node.
        return _CallGraphNodeType.LEAF

    # Base case 3: Empty list of callees
    if not callee_counts:
        return _CallGraphNodeType.LEAF_FROM_GENERALIZATION

    for callee, n in callee_counts:
        # Quite important: we do the recursive call first before adding in the edges.
        # Otherwise, adding the edge would mark the callee node as already-visited by
        # virtue of it being added to the graph with the `g.add_edge` call.

        # Do the recursive step, which will continue to mutate `g`
        callee_t = _build_call_graph(callee, generalizer, ssa, keep, max_depth, g, depth + 1)
        if callee_t == _CallGraphNodeType.LEAF:
            _add_costs(g, callee, costs=callee.my_leaf_costs())

        # Update edge in `g`
        if (bloq, callee) in g.edges:
            g.edges[bloq, callee]['n'] += n
        else:
            g.add_edge(bloq, callee, n=n)


def _sum_up_costs(g: nx.DiGraph) -> Dict[Bloq, Dict[CostKey, CostVal]]:
    """..."""
    costs: Dict[Bloq, Dict[CostKey, CostVal]] = {}
    for caller in reversed(list(nx.topological_sort(g))):
        bloq_costs: Dict[CostKey, CostVal] = {}

        # Accumulate contributions from callee costs, which have already been filled in
        # due to the reversed topological sort.
        for callee in g.successors(caller):
            n = g.edges[caller, callee]['n']
            existing_bloq_costs: Dict[CostKey, CostVal] = costs[callee]

            # caller_cost = sum(n * callee_cost)
            for k, v in existing_bloq_costs.items():
                if k not in bloq_costs:
                    bloq_costs[k] = k.identity_val()
                bloq_costs[k] += v * n

        # static costs
        for k, v in g.nodes[caller]['costs']:
            if k not in bloq_costs:
                bloq_costs[k] = v
            else:
                bloq_costs[k] += v

        costs[caller] = bloq_costs

    return costs


def _compute_sigma(root_bloq: Bloq, g: nx.DiGraph) -> Dict[Bloq, Union[int, sympy.Expr]]:
    costs = _sum_up_costs(g)
    sigma = {}
    for cost_key, cost_val in costs[root_bloq].items():
        if isinstance(cost_key, BloqCount):
            assert isinstance(cost_val, AddCostVal)
            sigma[cost_key.bloq] = cost_val.qty

    return sigma


def _make_composite_generalizer(*funcs: GeneralizerT) -> GeneralizerT:
    """Return a generalizer that calls each `*funcs` generalizers in order."""

    def _composite_generalize(b: Bloq) -> Optional[Bloq]:
        for func in funcs:
            b = func(b)
            if b is None:
                return
        return b

    return _composite_generalize


def get_bloq_call_graph(
    bloq: Bloq,
    generalizer: Optional[Union['GeneralizerT', Sequence['GeneralizerT']]] = None,
    ssa: Optional[SympySymbolAllocator] = None,
    keep: Optional[Callable[[Bloq], bool]] = None,
    max_depth: Optional[int] = None,
) -> Tuple[nx.DiGraph, Dict[Bloq, Union[int, sympy.Expr]]]:
    """Recursively build the bloq call graph and call totals.

    See `Bloq.call_graph()` as a convenient way of calling this function.

    Args:
        bloq: The bloq to count sub-bloqs.
        generalizer: If provided, run this function on each (sub)bloq to replace attributes
            that do not affect resource estimates with generic sympy symbols. If the function
            returns `None`, the bloq is omitted from the counts graph. If a sequence of
            generalizers is provided, each generalizer will be run in order.
        ssa: a `SympySymbolAllocator` that will be passed to the `Bloq.build_call_graph` method. If
            your `generalizer` function closes over a `SympySymbolAllocator`, provide it here as
            well. Otherwise, we will create a new allocator.
        keep: If this function evaluates to True for the current bloq, keep the bloq as a leaf
            node in the call graph instead of recursing into it.
        max_depth: If provided, build a call graph with at most this many layers.

    Returns:
        g: A directed graph where nodes are (generalized) bloqs and edge attribute 'n' reports
            the number of times successor bloq is called via its predecessor.
        sigma: Call totals for "leaf" bloqs. We keep a bloq as a leaf in the call graph
            according to `keep` and `max_depth` (if provided) or if a bloq cannot be
            decomposed.
    """
    if ssa is None:
        ssa = SympySymbolAllocator()
    if keep is None:
        keep = lambda b: False
    if generalizer is None:
        generalizer = lambda b: b
    if isinstance(generalizer, (list, tuple)):
        generalizer = _make_composite_generalizer(*generalizer)

    g = nx.DiGraph()
    bloq = generalizer(bloq)
    if bloq is None:
        raise ValueError("You can't generalize away the root bloq.")
    _build_call_graph(bloq, generalizer, ssa, keep, max_depth, g=g, depth=0)
    sigma = _compute_sigma(bloq, g)
    return g, sigma


def print_counts_graph(g: nx.DiGraph):
    """Print the graph returned from `get_bloq_counts_graph`."""
    for b in nx.topological_sort(g):
        for succ in g.succ[b]:
            print(b, '--', g.edges[b, succ]['n'], '->', succ)
