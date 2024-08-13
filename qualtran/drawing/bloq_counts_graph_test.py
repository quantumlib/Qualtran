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
import random
import re
from typing import List

import networkx as nx

from qualtran.bloqs.for_testing import TestBloqWithCallGraph
from qualtran.bloqs.mcmt.and_bloq import MultiAnd
from qualtran.drawing import (
    format_counts_graph_markdown,
    format_counts_sigma,
    GraphvizCallGraph,
    GraphvizCounts,
)
from qualtran.drawing.bloq_counts_graph import _CallGraphDrawerBase
from qualtran.resource_counting import get_bloq_call_graph


def test_format_counts_sigma():
    graph, sigma = get_bloq_call_graph(MultiAnd(cvs=(1,) * 6))
    ret = format_counts_sigma(sigma)
    assert (
        ret
        == """\
#### Counts totals:
 - `ArbitraryClifford(n=2)`: 45
 - `T`: 20"""
    )


def test_format_counts_graph_markdown():
    graph, sigma = get_bloq_call_graph(MultiAnd(cvs=(1,) * 6))
    ret = format_counts_graph_markdown(graph)
    assert (
        ret
        == """\
 - `MultiAnd(n=6)`
   - `And`: $\\displaystyle 5$
 - `And`
   - `ArbitraryClifford(n=2)`: $\\displaystyle 9$
   - `T`: $\\displaystyle 4$
"""
    )


def test_graphviz_counts():
    graph, sigma = get_bloq_call_graph(MultiAnd(cvs=(1,) * 6))
    gvc = GraphvizCounts(graph)

    # The main test is in the drawing notebook, so please spot check that.
    # Here: we make sure the edge labels are 5, 9 or 4 (see above)
    dot_lines = gvc.get_graph().to_string().splitlines()
    edge_lines = [line for line in dot_lines if '->' in line]
    for line in edge_lines:
        ma = re.search(r'label=(\w+)', line)
        assert ma is not None, line
        i = int(ma.group(1))
        assert i in [5, 9, 4]


def test_abbreviate_details():
    namevals = [('x', 5), ('y', 100), ('s', 'a' * 100), ('x1', 1.2), ('x2', 1.3), ('x3', 1.4)]

    assert (
        GraphvizCounts.abbreviate_field_list(namevals)
        == "x=5, y=100, s='aaaaaaa ..., x1=1.2, [2 addtl fields]."
    )
    assert (
        GraphvizCounts.abbreviate_field_list(namevals[:5])
        == "x=5, y=100, s='aaaaaaa ..., x1=1.2, x2=1.3"
    )


def _get_node_labels_from_pydot_graph(drawer: _CallGraphDrawerBase) -> List[str]:
    graph = drawer.get_graph()
    node_labels = [node.get_label() for node in graph.get_node_list()]
    random.shuffle(node_labels)  # don't rely on order of graphviz nodes
    return node_labels


def test_graphviz_call_graph_no_data():
    # Example call graph
    bloq = TestBloqWithCallGraph()
    call_graph, _ = bloq.call_graph()

    drawer = GraphvizCallGraph(call_graph)
    node_labels = _get_node_labels_from_pydot_graph(drawer)
    for nl in node_labels:
        # Spot check one of the nodes
        if 'TestBloqWithCallGraph' in nl:
            assert nl == (
                '<<font point-size="10"><table border="0" cellborder="1" cellspacing="0" cellpadding="5">\n'
                '<tr><td colspan="2"><font point-size="10">TestBloqWithCallGraph</font></td></tr>\n'
                '</table></font>>'
            )


def test_graphviz_call_graph_with_data():
    # Example call graph
    bloq = TestBloqWithCallGraph()
    call_graph, _ = bloq.call_graph()

    # Collect T-Complexity data
    bloq_data = {}
    for bloq in call_graph.nodes:
        tcomp = bloq.t_complexity()
        record = {'T count': tcomp.t, 'clifford': tcomp.clifford, 'rot': tcomp.rotations}
        bloq_data[bloq] = record

    drawer = GraphvizCallGraph(call_graph, bloq_data=bloq_data)
    node_labels = _get_node_labels_from_pydot_graph(drawer)
    for nl in node_labels:
        # Spot check one of the nodes
        if 'TestBloqWithCallGraph' in nl:
            assert nl == (
                '<<font point-size="10"><table border="0" cellborder="1" cellspacing="0" cellpadding="5">\n'
                '<tr><td colspan="2"><font point-size="10">TestBloqWithCallGraph</font></td></tr>\n'
                '<tr><td>T count</td><td>100*_n0 + 600</td></tr><tr><td>clifford</td><td>0</td></tr><tr><td>rot</td><td>0</td></tr></table></font>>'
            )


def test_graphviz_call_graph_from_bloq():
    bloq = TestBloqWithCallGraph()
    drawer = GraphvizCallGraph.from_bloq(bloq)

    node_labels = _get_node_labels_from_pydot_graph(drawer)
    for nl in node_labels:
        # Spot check one of the nodes
        if 'TestBloqWithCallGraph' in nl:
            assert nl == (
                '<<font point-size="10"><table border="0" cellborder="1" cellspacing="0" cellpadding="5">\n'
                '<tr><td colspan="2"><font point-size="10">TestBloqWithCallGraph</font></td></tr>\n'
                '<tr><td>Qubits</td><td>3</td></tr>'
                '<tr><td>Ts</td><td>100*_n0 + 600</td></tr>'
                '</table></font>>'
            )


def test_graphviz_call_graph_from_bloq_agg():
    bloq = TestBloqWithCallGraph()
    drawer = GraphvizCallGraph.from_bloq(bloq, agg_gate_counts='t_and_ccz')

    node_labels = _get_node_labels_from_pydot_graph(drawer)
    for nl in node_labels:
        # Spot check one of the nodes
        # Note the additional cell.
        if 'TestBloqWithCallGraph' in nl:
            assert nl == (
                '<<font point-size="10"><table border="0" cellborder="1" cellspacing="0" cellpadding="5">\n'
                '<tr><td colspan="2"><font point-size="10">TestBloqWithCallGraph</font></td></tr>\n'
                '<tr><td>Qubits</td><td>3</td></tr>'
                '<tr><td>Ts</td><td>100*_n0 + 600</td></tr>'
                '<tr><td>CCZs</td><td>0</td></tr>'
                '</table></font>>'
            )


def test_graphviz_call_graph_from_bloq_max_depth():
    bloq = TestBloqWithCallGraph()
    drawer = GraphvizCallGraph.from_bloq(bloq)
    assert len(list(nx.topological_generations(drawer.g))) == 3
    drawer2 = GraphvizCallGraph.from_bloq(bloq, max_depth=1)
    assert len(list(nx.topological_generations(drawer2.g))) == 2
