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

"""Classes for drawing bloq counts graphs with Graphviz."""

from typing import Dict, Union

import IPython.display
import networkx as nx
import pydot
import sympy

from qualtran import Bloq, CompositeBloq


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
        if isinstance(b, CompositeBloq):
            details = f'{len(b.bloq_instances)} bloqs...'
        else:
            details = repr(b)

        label = [
            '<',
            f'{b.pretty_name().replace("<", "&lt;").replace(">", "&gt;")}<br />',
            f'<font face="monospace" point-size="10">{details}</font><br/>',
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


def _format_bloq_expr_markdown(bloq: Bloq, expr: Union[int, sympy.Expr]) -> str:
    """Return "`bloq`: expr" as markdown."""
    try:
        expr = expr._repr_latex_()
    except AttributeError:
        expr = f'{expr}'

    return f'`{bloq}`: {expr}'


def format_counts_graph_markdown(graph: nx.DiGraph) -> str:
    """Format a text version of `graph` as markdown."""
    m = ""
    for bloq in nx.topological_sort(graph):
        if not graph.succ[bloq]:
            continue
        m += f' - `{bloq}`\n'

        succ_lines = []
        for succ in graph.succ[bloq]:
            expr = sympy.sympify(graph.edges[bloq, succ]['n'])
            succ_lines.append(f'   - {_format_bloq_expr_markdown(succ, expr)}\n')
        succ_lines.sort()
        m += ''.join(succ_lines)

    return m


def format_counts_sigma(sigma: Dict[Bloq, Union[int, sympy.Expr]]) -> str:
    """Format `sigma` as markdown."""
    lines = [f' - {_format_bloq_expr_markdown(bloq, expr)}' for bloq, expr in sigma.items()]
    lines.sort()
    lines.insert(0, '#### Counts totals:')
    return '\n'.join(lines)
