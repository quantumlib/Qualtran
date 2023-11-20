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

import html
from typing import Any, Dict, Iterable, Tuple, Union

import attrs
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

        self.max_detail_fields = 5
        self.max_field_val_len = 12
        self.max_detail_len = 200

    def get_id(self, b: Bloq) -> str:
        if b in self._ids:
            return self._ids[b]
        new_id = f'b{self._i}'
        self._i += 1
        self._ids[b] = new_id
        return new_id

    def get_node_title(self, b: Bloq):
        """Return text to use as a title of a node.

        Override this method for complete control over the titles of nodes.
        """
        return b.pretty_name()

    @staticmethod
    def abbreviate_field_list(
        name_vals: Iterable[Tuple[str, Any]], max_field_val_len: int = 12, max_detail_fields=5
    ):
        """Helper function for abbreviating a list of key=value representations.

        This is used by the default `get_node_details`.
        """

        def abbrev(x: str):
            if len(x) > max_field_val_len:
                return x[: max_field_val_len - 4] + ' ...'
            return x

        details = [f'{name}={abbrev(repr(val))}' for name, val in name_vals]
        if len(details) > max_detail_fields:
            n = len(details) - max_detail_fields + 1
            details = details[: max_detail_fields - 1] + [f'[{n} addtl fields].']

        return ', '.join(details)

    def get_node_details(self, b: Bloq):
        """Return text to use as details for a node.

        Override this method for complete control over the details attached to nodes.
        """
        if isinstance(b, CompositeBloq):
            return f'{len(b.bloq_instances)} bloqs...'[: self.max_detail_len]

        if not attrs.has(b.__class__):
            return repr(b)[: self.max_detail_len]

        return self.abbreviate_field_list(
            ((field.name, getattr(b, field.name)) for field in attrs.fields(b.__class__)),
            max_field_val_len=self.max_field_val_len,
            max_detail_fields=self.max_detail_fields,
        )[: self.max_detail_len]

    def get_node_properties(self, b: Bloq):
        """Get graphviz properties for a bloq node representing `b`."""
        title = self.get_node_title(b)
        details = self.get_node_details(b)

        title = html.escape(title)
        details = html.escape(details)

        label = ['<', title]
        if details:
            label += ['<br />', f'<font face="monospace" point-size="10">{details}</font><br/>']
        label += ['>']
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
