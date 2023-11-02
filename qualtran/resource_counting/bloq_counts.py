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
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import sympy

from qualtran import Bloq, CompositeBloq, DecomposeNotImplementedError, DecomposeTypeError

BloqCountT = Tuple[Bloq, Union[int, sympy.Expr]]


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


def _build_cbloq_counts_graph(cbloq: CompositeBloq) -> Set[BloqCountT]:
    """Count all the subbloqs in a composite bloq.

    `CompositeBloq.resource_counting` calls this with no generalizer.

    Args:
        cbloq: The composite bloq.
        generalizer: A function that replaces bloq attributes that do not affect resource costs
            with sympy placeholders.
    """

    counts: Dict[Bloq, int] = defaultdict(lambda: 0)
    for binst in cbloq.bloq_instances:
        counts[binst.bloq] += 1

    return {(bloq, n) for bloq, n in counts.items()}


def _generalize_callees(
    raw_callee_counts: Set[BloqCountT], generalizer: Callable[[Bloq], Optional[Bloq]]
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


def _build_call_graph_inner(
    bloq: Bloq,
    generalizer: Callable[[Bloq], Optional[Bloq]],
    ssa: SympySymbolAllocator,
    keep: Callable[[Bloq], bool],
    max_depth: Optional[int],
    g: nx.DiGraph,
    depth: int,
) -> Dict[Bloq, Union[int, sympy.Expr]]:
    """The inner logic of the recursive `_build_call_graph`.

    This raises `DecomposeTypeError` if we're not supposed to be recursing on `bloq`.
    This may also raise `DecomposeNotImplementedError` if the call to `bloq.build_call_graph`
    uses its default fallback (based on decomposition), and a decomposition is not implemented.

    The outer `_build_call_graph` function handles these exceptions with a universal
    "we're a leaf node" code path.
    """
    # Base case 1: This node is requested by the user to be a leaf node via the `keep` parameter.
    if keep(bloq):
        raise DecomposeTypeError("Requested via `keep`.")

    # Base case 2: Max depth exceeded
    if max_depth is not None and depth >= max_depth:
        raise DecomposeTypeError("Max depth exceeded.")

    # Prep for recursion: get the callees and modify them according to `generalizer`.
    callee_counts = _generalize_callees(bloq.build_call_graph(ssa), generalizer)

    # Base case 3: Empty list of callees
    if not callee_counts:
        raise DecomposeTypeError("Empty list of callees.")

    sigma: Dict[Bloq, Union[int, sympy.Expr]] = defaultdict(lambda: 0)
    for callee, n in callee_counts:
        # Quite important: we do the recursive call first before adding in the edges.
        # Otherwise, adding the edge would mark the callee node as already-visited by
        # virtue of it being added to the graph with the `g.add_edge` call.

        # Do the recursive step, which will continue to mutate `g`
        callee_sigma = _build_call_graph(callee, generalizer, ssa, keep, max_depth, g, depth + 1)

        # Update edge in `g`
        if (bloq, callee) in g.edges:
            g.edges[bloq, callee]['n'] += n
        else:
            g.add_edge(bloq, callee, n=n)

        # Update `sigma` with the recursion results.
        for k in callee_sigma.keys():
            sigma[k] += callee_sigma[k] * n

    return dict(sigma)


def _build_call_graph(
    bloq: Bloq,
    generalizer: Callable[[Bloq], Optional[Bloq]],
    ssa: SympySymbolAllocator,
    keep: Callable[[Bloq], bool],
    max_depth: Optional[int],
    g: nx.DiGraph,
    depth: int,
) -> Dict[Bloq, Union[int, sympy.Expr]]:
    """Recursive counting function.

    Arguments are the same as `get_bloq_call_graph`, except `g` is the graph we're building
    (i.e. it is mutated by this function) and `depth` is the current recursion depth.
    """
    if bloq in g:
        # We already visited this node.
        return {}

    # Make sure this node is present in the graph. You could annotate
    # additional node properties here, too.
    g.add_node(bloq)

    try:
        return _build_call_graph_inner(bloq, generalizer, ssa, keep, max_depth, g, depth)
    except (DecomposeNotImplementedError, DecomposeTypeError):
        return {bloq: 1}


def get_bloq_call_graph(
    bloq: Bloq,
    generalizer: Callable[[Bloq], Optional[Bloq]] = None,
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
            returns `None`, the bloq is omitted from the counts graph.
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

    g = nx.DiGraph()
    bloq = generalizer(bloq)
    if bloq is None:
        raise ValueError("You can't generalize away the root bloq.")
    sigma = _build_call_graph(bloq, generalizer, ssa, keep, max_depth, g=g, depth=0)
    return g, sigma


def print_counts_graph(g: nx.DiGraph):
    """Print the graph returned from `get_bloq_counts_graph`."""
    for b in nx.topological_sort(g):
        for succ in g.succ[b]:
            print(b, '--', g.edges[b, succ]['n'], '->', succ)
