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
import re

from qualtran.bloqs.and_bloq import MultiAnd
from qualtran.drawing import format_counts_graph_markdown, format_counts_sigma, GraphvizCounts
from qualtran.resource_counting import get_bloq_call_graph


def test_format_counts_sigma():
    graph, sigma = get_bloq_call_graph(MultiAnd(cvs=(1,) * 6))
    ret = format_counts_sigma(sigma)
    assert (
        ret
        == """\
#### Counts totals:
 - `ArbitraryClifford(n=2)`: 45
 - `TGate()`: 20"""
    )


def test_format_counts_graph_markdown():
    graph, sigma = get_bloq_call_graph(MultiAnd(cvs=(1,) * 6))
    ret = format_counts_graph_markdown(graph)
    assert (
        ret
        == r""" - `MultiAnd(cvs=(1, 1, 1, 1, 1, 1))`
   - `And(cv1=1, cv2=1, uncompute=False)`: $\displaystyle 5$
 - `And(cv1=1, cv2=1, uncompute=False)`
   - `ArbitraryClifford(n=2)`: $\displaystyle 9$
   - `TGate()`: $\displaystyle 4$
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
