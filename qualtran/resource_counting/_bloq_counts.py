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
from typing import Callable, Dict, Iterator, Mapping, Sequence, Tuple, TYPE_CHECKING

import attrs
import networkx as nx
import numpy as np
import sympy
from attrs import field, frozen

from qualtran.symbolics import ceil, log2, ssum, SymbolicFloat, SymbolicInt

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


FloatRepr_T = str
"""The type to represent floats as, to use as safe keys in mappings."""


def _mapping_to_counter(mapping: Mapping[FloatRepr_T, int]) -> Counter[FloatRepr_T]:
    if isinstance(mapping, Counter):
        return mapping
    return Counter(mapping)


@frozen(kw_only=True)
class GateCounts:
    """A data class of counts of the typical target gates in a compilation.

    Specifically, this class holds counts for the number of `TGate` (and adjoint), `Toffoli`,
    `TwoBitCSwap`, `And`, clifford bloqs, single qubit rotations, and measurements.
    """

    t: SymbolicInt = 0
    toffoli: SymbolicInt = 0
    cswap: SymbolicInt = 0
    and_bloq: SymbolicInt = 0
    clifford: SymbolicInt = 0
    measurement: SymbolicInt = 0
    binned_rotation_epsilons: Counter[FloatRepr_T] = field(
        factory=Counter, converter=_mapping_to_counter, eq=lambda d: tuple(d.items())
    )

    @classmethod
    def from_rotation_with_eps(cls, eps: float, *, n_rotations: int = 1, eps_repr_prec: int = 10):
        """Construct a GateCount with a rotation of precision `eps`.

        Args:
            eps: precision to synthesize the rotation(s).
            eps_repr_prec: number of digits to approximate `eps` to. Uses 10 by default.
                           See `np.format_float_scientific` for more details.
            n_rotations: number of rotations, defaults to 1.
        """
        eps_bin = np.format_float_scientific(eps, precision=eps_repr_prec, unique=False)
        return cls(binned_rotation_epsilons=Counter({eps_bin: n_rotations}))

    def iter_rotations_with_epsilon(self) -> Iterator[tuple[float, SymbolicInt]]:
        """Iterate through the rotation precisions (epsilon) and their frequency."""
        for eps_bin, n_rot in self.binned_rotation_epsilons.items():
            yield float(eps_bin), n_rot

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
            binned_rotation_epsilons=self.binned_rotation_epsilons + other.binned_rotation_epsilons,
        )

    def __mul__(self, other):
        return GateCounts(
            t=other * self.t,
            toffoli=other * self.toffoli,
            cswap=other * self.cswap,
            and_bloq=other * self.and_bloq,
            clifford=other * self.clifford,
            measurement=other * self.measurement,
            binned_rotation_epsilons=Counter(
                {eps_bin: other * n_rot for eps_bin, n_rot in self.binned_rotation_epsilons.items()}
            ),
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __str__(self):
        strs = [f'{k}: {v}' for k, v in self.asdict().items()]
        if strs:
            return ', '.join(strs)
        return '-'

    def asdict(self) -> Dict[str, int]:
        d = attrs.asdict(self)

        def _is_nonzero(v):
            maybe_nonzero = sympy.sympify(v)
            if maybe_nonzero is None:
                return True
            return maybe_nonzero

        def _keep(key, value) -> bool:
            if key == 'binned_rotation_epsilons':
                return value
            return _is_nonzero(value)

        return {k: v for k, v in d.items() if _keep(k, v)}

    @staticmethod
    def rotation_t_cost(eps: SymbolicFloat) -> SymbolicInt:
        """T-cost of a single Z rotation with precision `eps`.

        References:
            [Efficient synthesis of universal Repeat-Until-Success circuits](https://arxiv.org/abs/1404.5320)
            Bocharov et. al. 2014. Page 4, Paragraph "Simulation Results."
        """
        return ceil(1.149 * log2(1.0 / eps) + 9.2)

    def total_rotations_as_t(self) -> SymbolicInt:
        """Total number of T Gates for the rotations."""
        return ssum(
            n_rotations * self.rotation_t_cost(eps)
            for eps, n_rotations in self.iter_rotations_with_epsilon()
        )

    def total_t_count(
        self, ts_per_toffoli: int = 4, ts_per_cswap: int = 7, ts_per_and_bloq: int = 4
    ) -> int:
        """Get the total number of T Gates for the `GateCounts` object.

        This simply multiplies each gate type by its cost in terms of T gates, which is configurable
        via the arguments to this method.
        """
        return (
            self.t
            + ts_per_toffoli * self.toffoli
            + ts_per_cswap * self.cswap
            + ts_per_and_bloq * self.and_bloq
            + self.total_rotations_as_t()
        )

    def total_t_and_ccz_count(self) -> Dict[str, SymbolicInt]:
        n_ccz = self.toffoli + self.cswap + self.and_bloq
        n_t = self.t + self.total_rotations_as_t()
        return {'n_t': n_t, 'n_ccz': n_ccz}

    @property
    def rotations_ignoring_eps(self) -> SymbolicInt:
        """Total number of rotations, ignoring the individual precisions."""
        return ssum(self.binned_rotation_epsilons.values())

    def total_beverland_count(self) -> Dict[str, SymbolicInt]:
        r"""Counts used by Beverland. et. al. using notation from the reference.

         - $M_\mathrm{meas}$ is the number of measurements.
         - $M_R$ is the number of rotations.
         - $M_T$ is the number of T operations.
         - $3*M_mathrm{Tof}$ is the number of Toffoli operations.
         - $D_R$ is the number of layers containing at least one rotation. This can be smaller than
           the total number of non-Clifford layers since it excludes layers consisting only of T or
           Toffoli gates. Since we don't compile the 'layers' explicitly, we set this to be the
           number of rotations.

        Note: This costing method ignores the individual rotation precisions (`eps`).

        Reference:
            https://arxiv.org/abs/2211.07629.
            Equation D3.
        """
        toffoli = self.toffoli + self.and_bloq + self.cswap
        rotation = self.rotations_ignoring_eps
        return {
            'meas': self.measurement,
            'R': rotation,
            'T': self.t,
            'Tof': toffoli,
            'D_R': rotation,
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
            return GateCounts.from_rotation_with_eps(bloq.eps)

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
