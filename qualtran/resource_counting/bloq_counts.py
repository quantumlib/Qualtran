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

"""Functionality for the `Bloq.bloq_counts()` protocol."""

from collections import defaultdict
from typing import Callable, Dict, Optional, Sequence, Set, Tuple, Union

import IPython.display
import networkx as nx
import pydot
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


def get_cbloq_bloq_counts(
    cbloq: CompositeBloq, generalizer: Callable[[Bloq], Optional[Bloq]] = None
) -> Set[BloqCountT]:
    """Count all the subbloqs in a composite bloq.

    `CompositeBloq.resource_counting` calls this with no generalizer.

    Args:
        cbloq: The composite bloq.
        generalizer: A function that replaces bloq attributes that do not affect resource costs
            with sympy placeholders.
    """
    if generalizer is None:
        generalizer = lambda b: b

    counts: Dict[Bloq, int] = defaultdict(lambda: 0)
    for binst in cbloq.bloq_instances:
        bloq = binst.bloq
        bloq = generalizer(bloq)
        if bloq is None:
            continue

        counts[bloq] += 1

    return {(n, bloq) for bloq, n in counts.items()}


def _descend_counts(
    parent: Bloq,
    g: nx.DiGraph,
    ssa: SympySymbolAllocator,
    generalizer: Callable[[Bloq], Optional[Bloq]],
    keep: Sequence[Bloq],
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
    if parent in keep:
        return {parent: 1}
    try:
        count_decomp = parent.bloq_counts(ssa)
    except NotImplementedError:
        # Base case 2: Decomposition (or `bloq_counts`) is not implemented. This is left as a
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
        child_counts = _descend_counts(child, g, ssa, generalizer, keep)

        # Update `sigma` with the recursion results.
        for k in child_counts.keys():
            sigma[k] += child_counts[k] * n

    return dict(sigma)


def get_bloq_counts_graph(
    bloq: Bloq,
    generalizer: Callable[[Bloq], Optional[Bloq]] = None,
    ssa: Optional[SympySymbolAllocator] = None,
    keep: Optional[Sequence[Bloq]] = None,
) -> Tuple[nx.DiGraph, Dict[Bloq, Union[int, sympy.Expr]]]:
    """Recursively gather bloq counts.

    Args:
        bloq: The bloq to count sub-bloqs.
        generalizer: If provided, run this function on each (sub)bloq to replace attributes
            that do not affect resource estimates with generic sympy symbols. If this function
            returns `None`, the bloq is ommitted from the counts graph.
        ssa: a `SympySymbolAllocator` that will be passed to the `Bloq.bloq_counts` methods. If
            your `generalizer` function closes over a `SympySymbolAllocator`, provide it here as
            well. Otherwise, we will create a new allocator.
        keep: Stop recursing and keep these bloqs as leaf nodes in the counts graph. Otherwise,
            leaf nodes are those without a decomposition.

    Returns:
        g: A directed graph where nodes are (generalized) bloqs and edge attribute 'n' counts
            how many of the successor bloqs are used in the decomposition of the predecessor
            bloq(s).
        sigma: A mapping from leaf bloqs to their total counts.

    """
    if ssa is None:
        ssa = SympySymbolAllocator()
    if keep is None:
        keep = []
    if generalizer is None:
        generalizer = lambda b: b

    g = nx.DiGraph()
    bloq = generalizer(bloq)
    if bloq is None:
        raise ValueError("You can't generalize away the root bloq.")
    sigma = _descend_counts(bloq, g, ssa, generalizer, keep)
    return g, sigma


def print_counts_graph(g: nx.DiGraph):
    """Print the graph returned from `get_bloq_counts_graph`."""
    for b in nx.topological_sort(g):
        for succ in g.succ[b]:
            print(b, '--', g.edges[b, succ]['n'], '->', succ)


def markdown_bloq_expr(bloq: Bloq, expr: Union[int, sympy.Expr]):
    """Return "`bloq`: expr" as markdown."""
    try:
        expr = expr._repr_latex_()
    except AttributeError:
        expr = f'{expr}'

    return f'`{bloq}`: {expr}'


def markdown_counts_graph(graph: nx.DiGraph) -> IPython.display.Markdown:
    """Render the graph returned from `get_bloq_counts_graph` as markdown."""
    m = ""
    for bloq in nx.topological_sort(graph):
        if not graph.succ[bloq]:
            continue
        m += f' - `{bloq}`\n'
        for succ in graph.succ[bloq]:
            expr = sympy.sympify(graph.edges[bloq, succ]['n'])
            m += f'   - {markdown_bloq_expr(bloq, expr)}\n'

    return IPython.display.Markdown(m)


def markdown_counts_sigma(sigma: Dict[Bloq, Union[int, sympy.Expr]]) -> IPython.display.Markdown:
    lines = []
    for bloq, expr in sigma.items():
        lines.append(' - ' + markdown_bloq_expr(bloq, expr))
    return IPython.display.Markdown('\n'.join(lines))


class GraphvizCounts:
    """This class turns a bloqs count graph into Graphviz objects and drawings.

    Args:
        g: The counts graph.
    """

    def __init__(self, g: nx.DiGraph):
        self.g = g
        self._ids: Dict[Bloq, str] = {}
        self._i = 0

    def get_id(self, b: Bloq) -> str:
        if b in self._ids:
            return self._ids[b]
        new_id = f'b{self._i}'
        self._i += 1
        self._ids[b] = new_id
        return new_id

    def get_node_properties(self, b: Bloq):
        """Get graphviz properties for a bloq node representing `b`."""
        label = [
            '<',
            f'{b.pretty_name().replace("<", "&lt;").replace(">", "&gt;")}<br />',
            f'<font face="monospace" point-size="10">{repr(b)}</font><br/>',
            '>',
        ]
        return {'label': ''.join(label), 'shape': 'rect'}

    def add_nodes(self, graph: pydot.Graph):
        """Helper function to add nodes to the pydot graph."""
        b: Bloq
        for b in nx.topological_sort(self.g):
            graph.add_node(pydot.Node(self.get_id(b), **self.get_node_properties(b)))

    def add_edges(self, graph: pydot.Graph):
        """Helper function to add edges to the pydot graph."""
        for b1, b2 in self.g.edges:
            n = self.g.edges[b1, b2]['n']
            label = sympy.printing.pretty(n)
            graph.add_edge(pydot.Edge(self.get_id(b1), self.get_id(b2), label=label))

    def get_graph(self):
        """Get the pydot graph."""
        graph = pydot.Dot('counts', graph_type='digraph', rankdir='TB')
        self.add_nodes(graph)
        self.add_edges(graph)
        return graph

    def get_svg_bytes(self) -> bytes:
        """Get the SVG code (as bytes) for drawing the graph."""
        return self.get_graph().create_svg()

    def get_svg(self) -> IPython.display.SVG:
        """Get an IPython SVG object displaying the graph."""
        return IPython.display.SVG(self.get_svg_bytes())
