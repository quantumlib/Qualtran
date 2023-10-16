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
from typing import Callable, Dict, Optional, Set, Tuple, Union

import networkx as nx
import sympy

from qualtran import Bloq, CompositeBloq

BloqCountT = Tuple[Union[int, sympy.Expr], Bloq]


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

    return {(n, bloq) for bloq, n in counts.items()}


def _recurse_call_graph(
    parent: Bloq,
    g: nx.DiGraph,
    ssa: SympySymbolAllocator,
    generalizer: Callable[[Bloq], Optional[Bloq]],
    keep: Callable[[Bloq], bool],
    max_depth: Optional[int],
    depth: int,
) -> Dict[Bloq, Union[int, sympy.Expr]]:
    """Recursive counting function.

    Args:
        parent: The current, parent bloq.
        g: The networkx graph to which we will add the parent and its successors. This
            argument is mutated!
        ssa: A SympySymbolAllocator to help canonicalize bloqs.
        generalizer: A function that replaces bloq attributes that do not affect resource costs
            with sympy placeholders using `ssa`.
        keep: Use these bloqs as the base case leaf nodes. Otherwise, we stop whenever there's
            no decomposition, i.e. when `parent.bloq_counts` raises NotImplementedError

    Returns:
        sigma: A dictionary keyed by bloqs whose value is the running sum.
    """
    g.add_node(parent)

    # Base case 1: This node is requested by the user to be a leaf node via the `keep` parameter.
    if keep(parent):
        return {parent: 1}

    # Base case 2: Max depth exceeded
    if max_depth is not None and depth >= max_depth:
        return {parent: 1}

    try:
        count_decomp = parent.build_call_graph(ssa)
    except NotImplementedError:  # TODO: DecomposeNotImplementedError
        # Base case 3: Decomposition (or `bloq_counts`) is not implemented. This is left as a
        #              leaf node.
        return {parent: 1}

    sigma: Dict[Bloq, Union[int, sympy.Expr]] = defaultdict(lambda: 0)
    for n, child in count_decomp:
        child = generalizer(child)
        if child is None:
            continue

        # Update edge in `g`
        if (parent, child) in g.edges:
            g.edges[parent, child]['n'] += n
        else:
            g.add_edge(parent, child, n=n)

        # Do the recursive step, which will continue to mutate `g`
        child_counts = _recurse_call_graph(child, g, ssa, generalizer, keep, max_depth, depth + 1)

        # Update `sigma` with the recursion results.
        for k in child_counts.keys():
            sigma[k] += child_counts[k] * n

    return dict(sigma)


def get_bloq_call_graph(
    bloq: Bloq,
    generalizer: Callable[[Bloq], Optional[Bloq]] = None,
    ssa: Optional[SympySymbolAllocator] = None,
    keep: Optional[Callable[[Bloq], bool]] = None,
    max_depth: Optional[int] = None,
) -> Tuple[nx.DiGraph, Dict[Bloq, Union[int, sympy.Expr]]]:
    """Recursively build the bloq call graph.

    We stop recursing and keep a bloq as a leaf in the call graph if 1) `keep` is provided
    and evaluates to True on the given bloq, 2) `max_depth` is provided and recursing would
    exceed the maximum, or 3) if a bloq cannot be decomposed.

    Args:
        bloq: The bloq to count sub-bloqs.
        generalizer: If provided, run this function on each (sub)bloq to replace attributes
            that do not affect resource estimates with generic sympy symbols. If the function
            returns `None`, the bloq is omitted from the counts graph.
        ssa: a `SympySymbolAllocator` that will be passed to the `Bloq.bloq_counts` methods. If
            your `generalizer` function closes over a `SympySymbolAllocator`, provide it here as
            well. Otherwise, we will create a new allocator.
        keep: If this function evaluates to True for the current bloq, keep the bloq as a leaf
            node in the call graph and stop recursing.
        max_depth: If provided, stop recursing after the given depth.

    Returns:
        g: A directed graph where nodes are (generalized) bloqs and edge attribute 'n' counts
            how many of the successor bloqs are used in the decomposition of the predecessor
            bloq(s).
        sigma: A mapping from leaf bloqs to their total counts.

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
    sigma = _recurse_call_graph(bloq, g, ssa, generalizer, keep, max_depth, depth=0)
    return g, sigma


def print_counts_graph(g: nx.DiGraph):
    """Print the graph returned from `get_bloq_counts_graph`."""
    for b in nx.topological_sort(g):
        for succ in g.succ[b]:
            print(b, '--', g.edges[b, succ]['n'], '->', succ)
