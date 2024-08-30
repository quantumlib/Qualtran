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
import abc
import html
import warnings
from typing import Any, cast, Dict, Iterable, Mapping, Optional, Tuple, TYPE_CHECKING, Union

import attrs
import IPython.display
import networkx as nx
import pydot
import sympy

from qualtran import Bloq, CompositeBloq
from qualtran.symbolics import SymbolicInt

if TYPE_CHECKING:
    from qualtran.resource_counting import CostKey, CostValT, GateCounts


class _CallGraphDrawerBase(metaclass=abc.ABCMeta):
    def __init__(self, g: nx.DiGraph):
        self.g = g
        self._ids: Dict[Bloq, str] = {}
        self._i = 0

    def get_id(self, b: Bloq) -> str:
        """Return a unique string id for each bloq encountered in `self.g`.

        String ids are required for graphviz.
        """
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
        raise NotImplementedError()

    def get_node_details(self, b: Bloq):
        """Return text to use as details for a node.

        Override this method for complete control over the details attached to nodes.
        """
        raise NotImplementedError()

    def get_node_properties(self, b: Bloq):
        """Get graphviz properties for a bloq node representing `b`.

        By default, this will craft a label from `get_node_title` and `get_node_details`,
        and a rectangular node shape. Override this method to provide further customization.
        """
        title = html.escape(self.get_node_title(b))
        details = html.escape(self.get_node_details(b))

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

    def get_graph(self) -> pydot.Dot:
        """Get the pydot graph."""
        graph = pydot.Dot('counts', graph_type='digraph', rankdir='TB')
        self.add_nodes(graph)
        self.add_edges(graph)
        return graph

    def get_svg_bytes(self) -> bytes:
        """Get the SVG code (as bytes) for drawing the graph."""
        return self.get_graph().create(prog='dot', format='svg', encoding='utf-8')

    def get_svg(self) -> IPython.display.SVG:
        """Get an IPython SVG object displaying the graph."""
        return IPython.display.SVG(self.get_svg_bytes())


class GraphvizCounts(_CallGraphDrawerBase):
    """Draw a bloq call graphs using Graphviz.

    Each node is a bloq with a `bloq.pretty_name()` label and an automatically-determined
    "details" string based on the bloqs attributes. For non-attrs classes, classes with
    a large number of fields, or classes where the fields' string representations are long;
    the details string will be abbreviated.

    Each edge is labeled with the number of times the "caller" (predecessor) bloq calls the
    "callee" (successor) bloq.

    Args:
        g: The call graph, from e.g. `Bloq.call_graph()`.

    See Also:
        `qualtran.drawing.show_call_graph`, which uses this class under-the-hood.
    """

    def __init__(self, g: nx.DiGraph):
        super().__init__(g=g)

        self.max_detail_fields = 5
        self.max_field_val_len = 12
        self.max_detail_len = 200

    def get_node_title(self, b: Bloq):
        return b.pretty_name()

    @staticmethod
    def abbreviate_field_list(
        name_vals: Iterable[Tuple[str, Any]], max_field_val_len: int = 12, max_detail_fields=5
    ):
        """Helper function for abbreviating a list of key=value representations.

        This is used by the default `get_node_details`.
        """

        def abbrev(x: str):
            # Each field value gets cut off if it's too long.
            if len(x) > max_field_val_len:
                return x[: max_field_val_len - 4] + ' ...'
            return x

        details = [f'{name}={abbrev(repr(val))}' for name, val in name_vals]
        if len(details) > max_detail_fields:
            # Too many fields gets cut off.
            n = len(details) - max_detail_fields + 1
            details = details[: max_detail_fields - 1] + [f'[{n} addtl fields].']

        return ', '.join(details)

    def get_node_details(self, b: Bloq):
        # Special case for composite bloqs.
        if isinstance(b, CompositeBloq):
            return f'{len(b.bloq_instances)} bloqs...'[: self.max_detail_len]

        # Clumsy truncation if it's not an attrs class, since we can't easily inspect the fields.
        if not attrs.has(b.__class__):
            return repr(b)[: self.max_detail_len]

        # Otherwise, use the abbreviation function.
        return self.abbreviate_field_list(
            ((field.name, getattr(b, field.name)) for field in attrs.fields(b.__class__)),
            max_field_val_len=self.max_field_val_len,
            max_detail_fields=self.max_detail_fields,
        )[: self.max_detail_len]


class GraphvizCallGraph(_CallGraphDrawerBase):
    """Draw a bloq call graph using Graphviz with additional data.

    Each edge is labeled with the number of times the "caller" (predecessor) bloq calls the
    "callee" (successor) bloq.

    The constructor of this class assumes you have already generated the call graph as a networkx
    graph and constructed any associated data. See the factory method
    `GraphvizCallGraph.from_bloq()` to set up a call graph diagram from a bloq with sensible
    defaults.

    This class uses a bloq's `__str__` string to title the bloq. Arbitrary additional tabular
    data can be provided with `bloq_data`.

    This graph drawer is the successor to the `GraphvizCounts` existing drawer,
    and will replace `GraphvizCounts` when all bloqs have been migrated to use `__str__()`.

    Args:
        g: The call graph, from e.g. `Bloq.call_graph()`.
        bloq_data: A mapping from a bloq to a set of key, value pairs to include in a table
            in each node. The keys and values must support `str()`.
    """

    def __init__(self, g: nx.DiGraph, bloq_data: Optional[Dict['Bloq', Dict[Any, Any]]] = None):
        super().__init__(g)

        if bloq_data is None:
            bloq_data = {}

        self.bloq_data = bloq_data

    @classmethod
    def format_qubit_count(cls, val: SymbolicInt) -> Dict[str, str]:
        """Format `QubitCount` cost values as a string.

        Args:
            val: The qubit count value, which should be an integer

        Returns:
            A dictionary mapping a string cost name to a string cost value.
        """
        return {'Qubits': f'{val}'}

    @classmethod
    def format_qec_gates_cost(cls, val: 'GateCounts', agg: Optional[str] = None) -> Dict[str, str]:
        """Format `QECGatesCost` cost values as a string.

        Args:
            val: The qec gate costs value, which should be a `GateCounts` dataclass.
            agg: One of 'factored', 'total_t', 't_and_ccz', or 'beverland' to
                (optionally) aggregate the gate counts. If not specified, the 'factored'
                approach is used where each type of gate is counted individually. See the
                methods on `GateCounts` for more information.

        Returns:
            A dictionary mapping string cost names to string cost values.
        """
        labels = {
            't': 'Ts',
            'n_t': 'Ts',
            'toffoli': 'Toffolis',
            'n_ccz': 'CCZs',
            'cswap': 'CSwaps',
            'and_bloq': 'Ands',
            'clifford': 'Cliffords',
            'rotation': 'Rotations',
            'rotations': 'Rotations',
            'measurement': 'Measurements',
        }
        counts_dict: Mapping[str, SymbolicInt]
        if agg is None or agg == 'factored':
            counts_dict = val.asdict()
        elif agg == 'total_t':
            counts_dict = {'t': val.total_t_count()}
        elif agg == 't_and_ccz':
            counts_dict = val.total_t_and_ccz_count()
        elif agg == 'beverland':
            counts_dict = val.total_beverland_count()
        elif agg == 'legacy':
            counts_dict = val.to_legacy_t_complexity().asdict()
        else:
            raise ValueError(f"Unknown aggregation mode {agg}.")

        return {labels.get(gate_k, gate_k): f'{gate_v}' for gate_k, gate_v in counts_dict.items()}

    @classmethod
    def format_cost_data(
        cls,
        cost_data: Dict['Bloq', Dict['CostKey', 'CostValT']],
        agg_gate_counts: Optional[str] = None,
    ) -> Dict['Bloq', Dict[str, str]]:
        """Format `cost_data` as human-readable strings.

        Args:
            cost_data: The cost data, likely returned from a call to `query_costs()`. This
                class method will delegate to `format_qubit_count` and `format_qec_gates_cost`
                for `QubitCount` and `QECGatesCost` cost keys, respectively.
            agg_gate_counts: One of 'factored', 'total_t', 't_and_ccz', or 'beverland' to
                (optionally) aggregate the gate counts. If not specified, the 'factored'
                approach is used where each type of gate is counted individually. See the
                methods on `GateCounts` for more information.

        Returns:
            For each bloq key, a table of label/value pairs consisting of
            human-readable labels and formatted values.
        """
        from qualtran.resource_counting import GateCounts, QECGatesCost, QubitCount

        disp_data: Dict['Bloq', Dict[str, str]] = {}
        for bloq in cost_data.keys():
            bloq_disp: Dict[str, str] = {}
            for cost_key, cost_val in cost_data[bloq].items():
                if isinstance(cost_key, QubitCount):
                    bloq_disp |= cls.format_qubit_count(cast(SymbolicInt, cost_val))
                elif isinstance(cost_key, QECGatesCost):
                    assert isinstance(cost_val, GateCounts)
                    bloq_disp |= cls.format_qec_gates_cost(cost_val, agg=agg_gate_counts)
                else:
                    warnings.warn(f"Unknown cost key {cost_key}")
                    bloq_disp[str(cost_key)] = str(cost_val)

            disp_data[bloq] = bloq_disp
        return disp_data

    @classmethod
    def from_bloq(
        cls, bloq: Bloq, *, max_depth: Optional[int] = None, agg_gate_counts: Optional[str] = None
    ) -> 'GraphvizCallGraph':
        """Draw a bloq call graph.

        This factory method will generate a call graph from the bloq, query the `QECGatesCost`
        and `QubitCount` costs, format the cost data, and merge it with the call graph
        to create a call graph diagram with annotated costs.

        For additional customization, users can construct the call graph and bloq data themselves
        and use the normal constructor, or provide minor display customizations by
        overriding the `format_xxx` class methods.

        Args:
            bloq: The bloq from which we construct the call graph and query the costs.
            max_depth: The maximum depth (from the root bloq) of the call graph to draw. Note
                that the cost computations will walk the whole call graph, but only the nodes
                within this depth will be drawn.
            agg_gate_counts: One of 'factored', 'total_t', 't_and_ccz', or 'beverland' to
                (optionally) aggregate the gate counts. If not specified, the 'factored'
                approach is used where each type of gate is counted individually. See the
                methods on `GateCounts` for more information.

        Returns:
            A `GraphvizCallGraph` diagram-drawer, whose methods can be used to generate
            graphviz inputs or SVG diagrams.
        """
        from qualtran.resource_counting import QECGatesCost, QubitCount, query_costs

        call_graph, _ = bloq.call_graph(max_depth=max_depth)
        cost_data: Dict['Bloq', Dict[CostKey, Any]] = query_costs(
            bloq, [QubitCount(), QECGatesCost()]
        )
        formatted_cost_data = cls.format_cost_data(cost_data, agg_gate_counts=agg_gate_counts)
        return cls(g=call_graph, bloq_data=formatted_cost_data)

    def get_node_title(self, b: Bloq):
        return str(b)

    def get_node_properties(self, b: 'Bloq'):
        title = html.escape(self.get_node_title(b))

        label = ['<']
        label += ['<font point-size="10">']
        label += ['<table border="0" cellborder="1" cellspacing="0" cellpadding="5">\n']
        label += [f'<tr><td colspan="2"><font point-size="10">{title}</font></td></tr>\n']

        for k, v in self.bloq_data.get(b, {}).items():
            label += [f'<tr><td>{k}</td><td>{v}</td></tr>']

        label += ['</table></font>']
        label += ['>']
        return {'label': ''.join(label), 'shape': 'plaintext'}


def _format_bloq_expr_markdown(bloq: Bloq, expr: Union[int, sympy.Expr]) -> str:
    """Return "`bloq`: expr" as markdown."""
    if isinstance(expr, int):
        expr_str = str(expr)
    else:
        try:
            expr_str = expr._repr_latex_()
        except AttributeError:
            expr_str = f'{expr}'

    return f'`{bloq}`: {expr_str}'


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
