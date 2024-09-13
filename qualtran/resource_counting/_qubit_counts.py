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

import logging
from typing import Callable, Set

import networkx as nx
from attrs import frozen

from qualtran import Bloq, Connection, DanglingT, DecomposeNotImplementedError, DecomposeTypeError
from qualtran._infra.composite_bloq import _binst_to_cxns, CompositeBloq
from qualtran.symbolics import smax, SymbolicInt

from ._call_graph import get_bloq_callee_counts
from ._costing import CostKey

logger = logging.getLogger(__name__)


def _cbloq_max_width(
    binst_graph: nx.DiGraph, _bloq_max_width: Callable[[Bloq], SymbolicInt] = lambda b: 0
) -> SymbolicInt:
    """Get the maximum width of a composite bloq.

    Specifically, we treat each binst in series. The width at each inter-bloq time point
    is the sum of the bitsizes of all the connections that are "in play". The width at each
    during-a-binst time point is the sum of the binst width (which is provided by the
    `_bloq_max_width` callable) and the bystander connections that are "in play". The max
    width is the maximum over all the time points.

    If the dataflow graph has more than one connected component, we treat each component
    independently.
    """
    max_width: SymbolicInt = 0
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
                max_width = smax(max_width, during_size)

            # After the binst, its successor connections are 'in play'.
            in_play.update(succ_cxns)
            after_size = sum(s.shape for s in in_play)
            max_width = smax(max_width, after_size)

    return max_width


@frozen
class QubitCount(CostKey[SymbolicInt]):
    """A cost estimating the number of qubits required to implement a bloq.

    The number of qubits is bounded from below by the number of qubits implied by the signature.
    If a bloq has no callees, the size implied by the signature will be returned. Otherwise,
    this CostKey will try to compute the number of qubits by inspecting the decomposition.

    In the decomposition, each (sub)bloq is considered to be executed sequentially. The "width"
    of the circuit (i.e. the number of qubits) at each sequence point is the number of qubits
    required by the subbloq (computed recursively) plus any "bystander" idling wires.

    This is an estimate for the number of qubits required by an algorithm. Specifically:
     - Bloqs are assumed to be executed sequentially, minimizing the number of qubits potentially
       at the expense of greater circuit depth or execution time.
     - We do not consider "tetris-ing" subbloqs. In a decomposition, each subbloq is assumed
       to be using all of its qubits for the duration of its execution. This could potentially
       overestimate the total number of qubits.

    This Min-Max style estimate can provide a good balance between accuracy and scalability
    of the accounting. To fully account for each qubit and manage space-vs-time trade-offs,
    you must comprehensively decompose your algorithm to a `cirq.Circuit` of basic gates and
    use a `cirq.QubitManager` to manage trade-offs. This may be computationally expensive for
    large algorithms.
    """

    def compute(
        self, bloq: 'Bloq', get_callee_cost: Callable[['Bloq'], SymbolicInt]
    ) -> SymbolicInt:
        """Compute an estimate of the number of qubits used by `bloq`.

        See the class docstring for more information.
        """
        # Most accurate:
        # Compute the number of qubits ("width") from the bloq's decomposition. We forward
        # the `get_callee_cost` function so this can recurse into subbloqs.
        if isinstance(bloq, CompositeBloq):
            logger.info("Computing %s by the passed-in CompositeBloq", self)
            return _cbloq_max_width(bloq._binst_graph, get_callee_cost)
        try:
            cbloq = bloq.decompose_bloq()
            logger.info("Computing %s for %s from its decomposition", self, bloq)
            return _cbloq_max_width(cbloq._binst_graph, get_callee_cost)
        except (DecomposeNotImplementedError, DecomposeTypeError):
            pass
        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred when trying to compute {self} for {bloq}: {e}"
            ) from e

        # Fallback:
        # Use the simple maximum of callees and of this bloq's signature. If there
        # are no callees, this will be the number of qubits implied by the signature.
        # In any case, this strategy is likely an under-estimate of the qubit count.
        min_bloq_size = bloq.signature.n_qubits()
        callees = get_bloq_callee_counts(bloq)
        tot: int = min_bloq_size
        logger.info("Computing %s for %s from %d callee(s)", self, bloq, len(callees))
        for callee, n in callees:
            tot = smax(tot, get_callee_cost(callee))
        return tot

    def zero(self) -> SymbolicInt:
        """Zero cost is zero qubits."""
        return 0

    def __str__(self):
        return 'qubit count'
