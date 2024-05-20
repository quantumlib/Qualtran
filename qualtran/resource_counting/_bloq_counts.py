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
from collections import defaultdict
from typing import Callable, Dict, Optional, Tuple, TYPE_CHECKING

import attrs
import networkx as nx
from attrs import frozen

from ._call_graph import get_bloq_callee_counts
from ._costing import CostKey
from .classify_bloqs import bloq_is_clifford

if TYPE_CHECKING:
    from qualtran import Bloq

logger = logging.getLogger(__name__)

BloqCountDict = Dict['Bloq', int]


@frozen
class BloqCount(CostKey[BloqCountDict]):
    gateset_bloqs: Tuple['Bloq', ...]
    gateset_name: str

    @classmethod
    def for_gateset(cls, gateset_name: str):
        from qualtran.bloqs.basic_gates import TGate, Toffoli, TwoBitCSwap

        if gateset_name == 't':
            bloqs = (TGate(), TGate(is_adjoint=True))
        elif gateset_name == 't+tof':
            bloqs = (TGate(), TGate(is_adjoint=True), Toffoli())
        elif gateset_name == 't+tof+cswap':
            bloqs = (TGate(), TGate(is_adjoint=True), Toffoli(), TwoBitCSwap())
        else:
            raise ValueError(f"Unknown gateset name {gateset_name}")

        return cls(bloqs, gateset_name=gateset_name)

    @classmethod
    def for_call_graph_leaf_bloqs(cls, g: nx.DiGraph):
        leaf_bloqs = {node for node in g.nodes if not g.succ[node]}
        return cls(tuple(leaf_bloqs), gateset_name='leaf')

    def compute(
        self, bloq: 'Bloq', get_callee_cost: Callable[['Bloq'], BloqCountDict]
    ) -> BloqCountDict:
        if bloq in self.gateset_bloqs:
            logger.info("Computing %s: %s is in the target gateset.", self, bloq)
            return {bloq: 1}

        totals: BloqCountDict = defaultdict(lambda: 0)
        callees = get_bloq_callee_counts(bloq)
        logger.info("Computing %s for %s from %d callee(s)", self, bloq, len(callees))
        for callee, n_times_called in callees:
            callee_cost = get_callee_cost(callee)
            for gateset_bloq, count in callee_cost.items():
                totals[gateset_bloq] += n_times_called * count

        return dict(totals)

    def zero(self) -> BloqCountDict:
        return {}

    def __str__(self):
        return f'{self.gateset_name} counts'


@frozen(kw_only=True)
class GateCounts:
    t: int = 0
    toffoli: int = 0
    cswap: int = 0
    and_bloq: int = 0
    clifford: int = 0

    def __add__(self, other):
        if not isinstance(other, GateCounts):
            raise TypeError(f"Can only add other `GateCounts` objects, not {self}")

        return GateCounts(
            t=self.t + other.t,
            toffoli=self.toffoli + other.toffoli,
            cswap=self.cswap + other.cswap,
            and_bloq=self.and_bloq + other.and_bloq,
            clifford=self.clifford + other.clifford,
        )

    def __mul__(self, other):
        if not isinstance(other, int):
            raise TypeError(f"Can only multiply `GateCounts` objects by integers, not {self}")

        return GateCounts(
            t=other * self.t,
            toffoli=other * self.toffoli,
            cswap=other * self.cswap,
            and_bloq=other * self.and_bloq,
            clifford=other * self.clifford,
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __str__(self):
        strs = []
        for f in attrs.fields(self.__class__):
            val = getattr(self, f.name)
            if val != 0:
                strs.append(f'{f.name}: {val}')

        if strs:
            return ', '.join(strs)
        return '-'

    @property
    def total_n_magic(self):
        """The total number of magic states.

        This can be used as a rough proxy for total cost. It is the sum of all the attributes
        other than `clifford`.
        """
        return self.t + self.toffoli + self.cswap + self.and_bloq

    @property
    def total_n_magic(self):
        """The total number of magic states.

        This can be used as a rough proxy for total cost. It is the sum of all the attributes
        other than `clifford`.
        """
        return self.t + self.toffoli + self.cswap + self.and_bloq


@frozen
class QECGatesCost(CostKey[GateCounts]):
    """Counts specifically for 'expensive' gates in a surface code error correction scheme."""

    ts_per_toffoli: Optional[int] = None
    toffolis_per_and: Optional[int] = None
    ts_per_and: Optional[int] = None
    toffolis_per_cswap: Optional[int] = None
    ts_per_cswap: Optional[int] = None

    def compute(self, bloq: 'Bloq', get_callee_cost: Callable[['Bloq'], GateCounts]) -> GateCounts:
        from qualtran.bloqs.basic_gates import TGate, Toffoli, TwoBitCSwap
        from qualtran.bloqs.mcmt.and_bloq import And

        # T gates
        if isinstance(bloq, TGate):
            return GateCounts(t=1)

        # Toffolis
        if isinstance(bloq, Toffoli):
            if self.ts_per_toffoli is not None:
                return GateCounts(t=self.ts_per_toffoli)
            else:
                return GateCounts(toffoli=1)

        # 'And' bloqs
        if isinstance(bloq, And) and not bloq.uncompute:
            if self.toffolis_per_and is not None:
                return GateCounts(toffoli=self.toffolis_per_and * self.ts_per_toffoli)
            elif self.ts_per_and is not None:
                return GateCounts(t=self.ts_per_and)
            else:
                return GateCounts(and_bloq=1)

        # CSwaps aka Fredkin
        if isinstance(bloq, TwoBitCSwap):
            if self.toffolis_per_cswap is not None:
                return GateCounts(toffoli=self.toffolis_per_cswap)
            elif self.ts_per_cswap is not None:
                return GateCounts(t=self.ts_per_cswap)
            else:
                return GateCounts(cswap=1)

        # Cliffords
        if bloq_is_clifford(bloq):
            return GateCounts(clifford=1)

        # Recursive case
        totals = GateCounts()
        callees = get_bloq_callee_counts(bloq)
        logger.info("Computing %s for %s from %d callee(s)", self, bloq, len(callees))
        for callee, n_times_called in callees:
            callee_cost = get_callee_cost(callee)
            totals += n_times_called * callee_cost
        return totals

    def zero(self) -> GateCounts:
        return GateCounts()

    def validate_val(self, val: GateCounts):
        if not isinstance(val, GateCounts):
            raise TypeError(f"{self} values should be `GateCounts`, got {val}")

    def __str__(self):
        gates = ['t']
        if self.ts_per_toffoli is None:
            gates.append('tof')
        if self.toffolis_per_and is None and self.ts_per_and is None:
            gates.append('and')
        if self.toffolis_per_cswap is None and self.ts_per_cswap is None:
            gates.append('cswap')
        return ','.join(gates) + ' counts'
