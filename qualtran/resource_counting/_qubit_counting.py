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

from typing import Callable, Set, Union

import networkx as nx
import sympy

from qualtran import Bloq, Connection, DanglingT
from qualtran._infra.composite_bloq import _binst_to_cxns


def _cbloq_max_width(
    binst_graph: nx.DiGraph, _bloq_max_width: Callable[[Bloq], int] = lambda b: 0
) -> Union[int, sympy.Expr]:
    """Get the maximum width of a composite bloq.

    Specifically, we treat each binst in series. The width at each inter-bloq time point
    is the sum of the bitsizes of all the connections that are "in play". The width at each
    during-a-binst time point is the sum of the binst width (which is provided by the
    `_bloq_max_width` callable) and the bystander connections that are "in play". The max
    width is the maximum over all the time points.

    If the dataflow graph has more than one connected component, we treat each component
    independently.
    """
    max_width: Union[int, sympy.Expr] = 0
    in_play: Set[Connection] = set()

    for cc in nx.weakly_connected_components(binst_graph):
        for binst in nx.topological_sort(binst_graph.subgraph(cc)):
            pred_cxns, succ_cxns = _binst_to_cxns(binst, binst_graph=binst_graph)

            # Remove inbound connections from those that are 'in play'.
            for cxn in pred_cxns:
                in_play.remove(cxn)

            if not isinstance(binst, DanglingT):
                # During the application of the binst, we have "observer" connections that have
                # width as well as the width from the binst itself. We consider the case where
                # the bloq may have a max_width greater than the max of its left/right registers.
                during_size = _bloq_max_width(binst.bloq) + sum(s.shape for s in in_play)
                max_width = sympy.Max(max_width, during_size)

            # After the binst, its successor connections are 'in play'.
            in_play.update(succ_cxns)
            after_size = sum(s.shape for s in in_play)
            max_width = sympy.Max(max_width, after_size)

    return max_width
