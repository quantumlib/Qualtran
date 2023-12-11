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

"""Classes for drawing classical data flows with Graphviz."""

from typing import Dict, TYPE_CHECKING

import pydot

from qualtran import Bloq, Connection

from .graphviz import PrettyGraphDrawer

if TYPE_CHECKING:
    from qualtran.simulation.classical_sim import ClassicalValT


class ClassicalSimGraphDrawer(PrettyGraphDrawer):
    """A graph drawer that labels each edge with a classical value.

    The (composite) bloq must be composed entirely of classically-simulable bloqs.

    Args:
        bloq: The (composite) bloq to draw.
        vals: Input classical values to propogate through the composite bloq.
    """

    def __init__(self, bloq: Bloq, vals: Dict[str, 'ClassicalValT']):
        super().__init__(bloq=bloq)
        from qualtran.simulation.classical_sim import call_cbloq_classically

        _, soq_assign = call_cbloq_classically(
            self._cbloq.signature, vals, self._cbloq._binst_graph
        )
        self._soq_assign = soq_assign

    def cxn_label(self, cxn: Connection) -> str:
        """Label the connection with its classical value."""
        # Thru registers share the same soquet
        # key in `soq_assign` for a bloq's left and right ports.
        # The value in `soq_assign` will be for the right, output
        # value. So we need `cxn.left` as the correct connection label.
        return str(self._soq_assign[cxn.left])

    def cxn_edge(self, left_id: str, right_id: str, cxn: Connection) -> pydot.Edge:
        return pydot.Edge(
            left_id,
            right_id,
            label=self.cxn_label(cxn),
            labelfloat=True,
            fontsize=10,
            fontcolor='darkblue',
            arrowhead='dot',
            arrowsize=0.25,
        )
