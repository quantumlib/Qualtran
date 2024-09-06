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
import warnings
from collections import defaultdict
from typing import Callable, cast, Dict, Sequence, Tuple, TYPE_CHECKING

import attrs
import networkx as nx
import sympy
from attrs import field, frozen

from qualtran.symbolics import is_zero, SymbolicInt

from ._call_graph import get_bloq_callee_counts
from ._costing import CostKey
from .classify_bloqs import (
    bloq_is_clifford,
    bloq_is_rotation,
    bloq_is_state_or_effect,
    bloq_is_t_like,
)

if TYPE_CHECKING:
    from qualtran import Bloq
    from qualtran.cirq_interop.t_complexity_protocol import TComplexity

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
    rotation: SymbolicInt = 0
    measurement: SymbolicInt = 0

    def __add__(self, other):
        if not isinstance(other, GateCounts):
            raise TypeError(f"Can only add other `GateCounts` objects, not {self}")

        return GateCounts(
            t=self.t + other.t,
            toffoli=self.toffoli + other.toffoli,
            cswap=self.cswap + other.cswap,
            and_bloq=self.and_bloq + other.and_bloq,
            clifford=self.clifford + other.clifford,
            rotation=self.rotation + other.rotation,
            measurement=self.measurement + other.measurement,
        )

    def __mul__(self, other):
        return GateCounts(
            t=other * self.t,
            toffoli=other * self.toffoli,
            cswap=other * self.cswap,
            and_bloq=other * self.and_bloq,
            clifford=other * self.clifford,
            rotation=other * self.rotation,
            measurement=other * self.measurement,
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

        return {k: v for k, v in d.items() if _is_nonzero(v)}

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

    def total_t_and_ccz_count(self, ts_per_rotation: int = 11) -> Dict[str, SymbolicInt]:
        n_ccz = self.toffoli + self.cswap + self.and_bloq
        n_t = self.t + ts_per_rotation * self.rotation
        return {'n_t': n_t, 'n_ccz': n_ccz}

    def total_toffoli_only(self) -> int:
        """The number of Toffoli-like gates, and raise an exception if there are Ts/rotations."""
        if not is_zero(self.t):
            raise ValueError(f"{self} contains T counts.")
        if not is_zero(self.rotation):
            raise ValueError(f"{self} contains rotations.")
        return self.toffoli + self.cswap + self.and_bloq

    def to_legacy_t_complexity(
        self,
        ts_per_toffoli: int = 4,
        ts_per_cswap: int = 7,
        ts_per_and_bloq: int = 4,
        cliffords_per_and_bloq: int = 9,
        cliffords_per_cswap: int = 10,
    ) -> 'TComplexity':
        """Return a legacy `TComplexity` object.

        This coalesces all the gate types into t, rotations, and clifford fields. The conversion
        factors can be tweaked using the arguments to this method.

        The argument `cliffords_per_and_bloq` sets the base number of clifford gates to
        add per `self.and_bloq`. To fully match the exact legacy `t_complexity` numbers, you
        must enable `QECGatesCost(legacy_shims=True)`, which will enable a shim that directly
        adds on clifford counts for the X-gates used to invert the And control lines.
        """
        from qualtran.cirq_interop.t_complexity_protocol import TComplexity

        return TComplexity(
            t=self.t
            + ts_per_toffoli * self.toffoli
            + ts_per_cswap * self.cswap
            + ts_per_and_bloq * self.and_bloq,
            rotations=cast(int, self.rotation),
            clifford=self.clifford
            + self.measurement
            + cliffords_per_and_bloq * self.and_bloq
            + cliffords_per_cswap * self.cswap,
        )

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


@frozen(kw_only=True)
class QECGatesCost(CostKey[GateCounts]):
    """Counts specifically for 'expensive' gates in a surface code error correction scheme.

    The cost value type for this CostKey is `GateCounts`.

    Args:
        legacy_shims: If enabled, modify the counting logic to match the peculiarities of
            the legacy `t_complexity` protocol.
    """

    legacy_shims: bool = False

    def compute(self, bloq: 'Bloq', get_callee_cost: Callable[['Bloq'], GateCounts]) -> GateCounts:
        from qualtran.bloqs.basic_gates import GlobalPhase, Identity, Toffoli, TwoBitCSwap
        from qualtran.bloqs.basic_gates._shims import Measure
        from qualtran.bloqs.bookkeeping._bookkeeping_bloq import _BookkeepingBloq
        from qualtran.bloqs.mcmt import And, MultiTargetCNOT

        if self.legacy_shims:
            legacy_val = bloq._t_complexity_()
            if legacy_val is not NotImplemented:
                warnings.warn(
                    "Please migrate explicit cost annotations to the general "
                    "`Bloq.my_static_costs` method override.",
                    DeprecationWarning,
                )
                return GateCounts(
                    t=legacy_val.t, clifford=legacy_val.clifford, rotation=legacy_val.rotations
                )

        # T gates
        if bloq_is_t_like(bloq):
            return GateCounts(t=1)

        # Toffolis
        if isinstance(bloq, Toffoli):
            return GateCounts(toffoli=1)

        # Measurement
        if isinstance(bloq, Measure):
            return GateCounts(measurement=1)

        # 'And' bloqs
        if isinstance(bloq, And):
            # To match the legacy `t_complexity` protocol, we can hack in the explicit
            # counts for the clifford operations used to invert the control bit.
            # Note: we *only* add in the clifford operations that correspond to correctly
            # setting the control line. The other clifford operations inherent in compiling
            # an And gate to the gateset considered by the legacy `t_complexity` protocol can be
            # simply added in as part of `GateCounts.to_legacy_t_complexity()`
            n_inverted_controls = (bloq.cv1 == 0) + int(bloq.cv2 == 0)
            if bloq.uncompute:
                if self.legacy_shims:
                    return GateCounts(clifford=3 + 2 * n_inverted_controls, measurement=1)
                else:
                    return GateCounts(measurement=1, clifford=1)

            if self.legacy_shims:
                return GateCounts(and_bloq=1, clifford=2 * n_inverted_controls)
            else:
                return GateCounts(and_bloq=1)

        # CSwaps aka Fredkin
        if isinstance(bloq, TwoBitCSwap):
            return GateCounts(cswap=1)

        if isinstance(bloq, MultiTargetCNOT):
            # TODO(https://github.com/quantumlib/Qualtran/issues/1318): Decide how to count this.
            if self.legacy_shims:
                # Legacy mode: don't treat this as one clifford. Use its decomposition.
                pass  # fall through
            else:
                return GateCounts(clifford=1)

        # Cliffords
        if bloq_is_clifford(bloq):
            return GateCounts(clifford=1)

        # States and effects
        if bloq_is_state_or_effect(bloq):
            return GateCounts()

        # Bookkeeping, empty bloqs
        if isinstance(bloq, _BookkeepingBloq) or isinstance(bloq, (GlobalPhase, Identity)):
            return GateCounts()

        if bloq_is_rotation(bloq):
            return GateCounts(rotation=1)

        # Recursive case
        totals = GateCounts()
        callees = get_bloq_callee_counts(bloq, ignore_decomp_failure=False)
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
