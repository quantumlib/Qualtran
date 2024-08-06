#  Copyright 2024 Google LLC
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
from collections import Counter, defaultdict
from typing import Callable, Dict, Mapping, Sequence, Tuple, TYPE_CHECKING

import attrs
import networkx as nx
from attrs import field, frozen

from ._call_graph import get_bloq_callee_counts
from ._costing import CostKey
from .classify_bloqs import bloq_is_clifford, bloq_is_rotation

if TYPE_CHECKING:
    from qualtran import Bloq

logger = logging.getLogger(__name__)

BloqCountDict = Dict['Bloq', int]


def _gateset_bloqs_to_tuple(bloqs: Sequence['Bloq']) -> Tuple['Bloq', ...]:
    return tuple(bloqs)


@frozen
class BloqCount(CostKey[BloqCountDict]):
    """A cost which is the count of a specific set of bloqs forming a gateset.

    Often, we wish to know the number of specific gates in our algorithm. This is a generic
    CostKey that can count any gate (bloq) of interest.

    The cost value type for this cost is a mapping from bloq to its count.

    Args:
        gateset_bloqs: A sequence of bloqs which we will count. Bloqs are counted according
            to their equality operator.
        gateset_name: A string name of the gateset. Used for display and debugging purposes.
    """

    gateset_bloqs: Sequence['Bloq'] = field(converter=_gateset_bloqs_to_tuple)
    gateset_name: str

    @classmethod
    def for_gateset(cls, gateset_name: str):
        """Helper constructor to configure this cost for some common gatesets.

        Args:
            gateset_name: One of 't', 't+tof', 't+tof+cswap'. This will construct a
                `BloqCount` cost with the indicated gates as the `gateset_bloqs`. In all
                cases, both TGate and its adjoint are included.
        """
        from qualtran.bloqs.basic_gates import TGate, Toffoli, TwoBitCSwap

        bloqs: Tuple['Bloq', ...]
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
        """Helper constructor to configure this cost for 'leaf' bloqs in a given call graph.

        Args:
            g: The call graph. Its leaves will be used for `gateset_bloqs`. This call graph
                can be generated from `Bloq.call_graph()`
        """
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
        # The additive identity of the bloq counts dictionary is an empty dictionary.
        return {}

    def __str__(self):
        return f'{self.gateset_name} counts'


def _mapping_to_counter(mapping: Mapping[float, int]) -> Counter[float]:
    if isinstance(mapping, Counter):
        return mapping
    return Counter(mapping)


@frozen(kw_only=True)
class GateCounts:
    """A data class of counts of the typical target gates in a compilation.

    Specifically, this class holds counts for the number of `TGate` (and adjoint), `Toffoli`,
    `TwoBitCSwap`, `And`, clifford bloqs, single qubit rotations, and measurements.
    """

    t: int = 0
    toffoli: int = 0
    cswap: int = 0
    and_bloq: int = 0
    clifford: int = 0
    measurement: int = 0
    rotation_epsilons: Counter[float] = field(factory=Counter, converter=_mapping_to_counter)

    @property
    def rotation(self):
        from qualtran.cirq_interop.t_complexity_protocol import TComplexity

        return sum(
            n_rotations * int(TComplexity.rotation_cost(eps))
            for eps, n_rotations in self.rotation_epsilons.items()
        )

    def __add__(self, other):
        if not isinstance(other, GateCounts):
            raise TypeError(f"Can only add other `GateCounts` objects, not {self}")

        return GateCounts(
            t=self.t + other.t,
            toffoli=self.toffoli + other.toffoli,
            cswap=self.cswap + other.cswap,
            and_bloq=self.and_bloq + other.and_bloq,
            clifford=self.clifford + other.clifford,
            measurement=self.measurement + other.measurement,
            rotation_epsilons=self.rotation_epsilons + other.rotation_epsilons,
        )

    def __mul__(self, other):
        return GateCounts(
            t=other * self.t,
            toffoli=other * self.toffoli,
            cswap=other * self.cswap,
            and_bloq=other * self.and_bloq,
            clifford=other * self.clifford,
            measurement=other * self.measurement,
            rotation_epsilons=Counter({k: other * v for k, v in self.rotation_epsilons.items()}),
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __str__(self):
        strs = []
        for k, v in self.asdict().items():
            strs.append(f'{k}: {v}')

        if strs:
            return ', '.join(strs)
        return '-'

    def asdict(self):
        d = attrs.asdict(self)

        def _keep(key, value) -> bool:
            if key == 'rotation_epsilons':
                return value
            return value > 0

        return {k: v for k, v in d.items() if _keep(k, v)}

    def total_t_count(
        self,
        ts_per_toffoli: int = 4,
        ts_per_cswap: int = 7,
        ts_per_and_bloq: int = 4,
        ts_per_rotation: int = 11,
    ) -> int:
        """Get the total number of T Gates for the `GateCounts` object.

        This simply multiplies each gate type by its cost in terms of T gates, which is configurable
        via the arguments to this method.

        The default value for `ts_per_rotation` assumes the rotation is approximated using
        `Mixed fallback` protocol with error budget 1e-3.
        """
        return (
            self.t
            + ts_per_toffoli * self.toffoli
            + ts_per_cswap * self.cswap
            + ts_per_and_bloq * self.and_bloq
            + ts_per_rotation * self.rotation
        )

    def total_t_and_ccz_count(self, ts_per_rotation: int = 11) -> Dict[str, int]:
        n_ccz = self.toffoli + self.cswap + self.and_bloq
        n_t = self.t + ts_per_rotation * self.rotation
        return {'n_t': n_t, 'n_ccz': n_ccz}

    def total_beverland_count(self) -> Dict[str, int]:
        r"""Counts used by Beverland. et. al. using notation from the reference.

         - $M_\mathrm{meas}$ is the number of measurements.
         - $M_R$ is the number of rotations.
         - $M_T$ is the number of T operations.
         - $3*M_mathrm{Tof}$ is the number of Toffoli operations.
         - $D_R$ is the number of layers containing at least one rotation. This can be smaller than
           the total number of non-Clifford layers since it excludes layers consisting only of T or
           Toffoli gates. Since we don't compile the 'layers' explicitly, we set this to be the
           number of rotations.

        Reference:
            https://arxiv.org/abs/2211.07629.
            Equation D3.
        """
        toffoli = self.toffoli + self.and_bloq + self.cswap
        return {
            'meas': self.measurement,
            'R': self.rotation,
            'T': self.t,
            'Tof': toffoli,
            'D_R': self.rotation,
        }


@frozen
class QECGatesCost(CostKey[GateCounts]):
    """Counts specifically for 'expensive' gates in a surface code error correction scheme.

    The cost value type for this CostKey is `GateCounts`.
    """

    def compute(self, bloq: 'Bloq', get_callee_cost: Callable[['Bloq'], GateCounts]) -> GateCounts:
        from qualtran.bloqs.basic_gates import TGate, Toffoli, TwoBitCSwap
        from qualtran.bloqs.basic_gates.rotation import _HasEps
        from qualtran.bloqs.mcmt.and_bloq import And

        # T gates
        if isinstance(bloq, TGate):
            return GateCounts(t=1)

        # Toffolis
        if isinstance(bloq, Toffoli):
            return GateCounts(toffoli=1)

        # 'And' bloqs
        if isinstance(bloq, And):
            if bloq.uncompute:
                return GateCounts(measurement=1, clifford=1)
            return GateCounts(and_bloq=1)

        # CSwaps aka Fredkin
        if isinstance(bloq, TwoBitCSwap):
            return GateCounts(cswap=1)

        # Cliffords
        if bloq_is_clifford(bloq):
            return GateCounts(clifford=1)

        if bloq_is_rotation(bloq):
            assert isinstance(bloq, _HasEps)
            return GateCounts(rotation_epsilons={bloq.eps: 1})

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
        return 'gate counts'
