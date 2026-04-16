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

"""Gate counting for the FLASQ cost model.

Walks a Qualtran bloq decomposition tree and tallies primitive gates into
FLASQGateCounts. Distance-dependent costs (span) are handled separately
by span_counting.py.
"""

import logging
from typing import Callable, Dict, Mapping, Optional, Tuple, Union

import attrs
import cirq
import numpy as np
import sympy
from attrs import frozen
from frozendict import frozendict

from qualtran import Bloq
from qualtran.bloqs.basic_gates import (
    CNOT,
    CZ,
    Hadamard,
    Rx,
    Rz,
    SGate,
    Toffoli,
    XGate,
    XPowGate,
    YGate,
    YPowGate,
    ZGate,
    ZPowGate,
)
from qualtran.bloqs.basic_gates.global_phase import GlobalPhase
from qualtran.bloqs.basic_gates.identity import Identity
from qualtran.bloqs.basic_gates.x_basis import MeasureX
from qualtran.bloqs.basic_gates.z_basis import MeasureZ
from qualtran.bloqs.bookkeeping._bookkeeping_bloq import _BookkeepingBloq
from qualtran.bloqs.mcmt import And
from qualtran.cirq_interop import CirqGateAsBloq
from qualtran.resource_counting import CostKey, get_bloq_callee_counts
from qualtran.resource_counting.classify_bloqs import (
    bloq_is_clifford,
    bloq_is_rotation,
    bloq_is_state_or_effect,
    bloq_is_t_like,
)
from qualtran.surface_code.flasq.utils import _to_frozendict
from qualtran.symbolics import is_zero, SymbolicFloat
from qualtran.symbolics.types import SymbolicInt

logger = logging.getLogger(__name__)


@frozen(kw_only=True)
class FLASQGateCounts:
    """Tallies of each primitive gate type encountered during circuit decomposition.

    Attributes:
        t: Number of T (or T†) gates.
        toffoli: Number of Toffoli gates.
        z_rotation: Number of R_Z(θ) rotation gates (non-Clifford angles).
        x_rotation: Number of R_X(θ) rotation gates (non-Clifford angles).
        hadamard: Number of Hadamard gates.
        s_gate: Number of S gates.
        cnot: Number of CNOT gates.
        cz: Number of CZ gates.
        and_gate: Number of And gates (forward direction).
        and_dagger_gate: Number of And† gates. Tracked separately because And†
            uses measurement-based uncomputation with cheaper ancilla volume.
        bloqs_with_unknown_cost: Bloqs the cost model does not recognize.
            Unrecognized Cliffords are flagged here rather than assumed free,
            so users can audit coverage of new circuits.

    Geometric information (span) is handled separately by ``span_counting``.
    """

    t: SymbolicFloat = 0
    toffoli: SymbolicFloat = 0
    z_rotation: SymbolicFloat = 0
    x_rotation: SymbolicFloat = 0
    hadamard: SymbolicFloat = 0
    s_gate: SymbolicFloat = 0
    cnot: SymbolicFloat = 0
    cz: SymbolicFloat = 0
    and_gate: SymbolicFloat = 0
    and_dagger_gate: SymbolicFloat = 0
    bloqs_with_unknown_cost: Mapping[Bloq, SymbolicInt] = attrs.field(
        converter=_to_frozendict, default=frozendict()
    )

    def __add__(self, other):
        if not isinstance(other, FLASQGateCounts):
            if isinstance(other, int) and other == 0:
                return self
            raise TypeError(
                f"Can only add other `FLASQGateCounts` objects or 0, not {type(other)}: {other}"
            )
        merged_unknowns = dict(self.bloqs_with_unknown_cost)
        for bloq, count in other.bloqs_with_unknown_cost.items():
            merged_unknowns[bloq] = merged_unknowns.get(bloq, 0) + count

        return FLASQGateCounts(
            t=self.t + other.t,
            toffoli=self.toffoli + other.toffoli,
            z_rotation=self.z_rotation + other.z_rotation,
            x_rotation=self.x_rotation + other.x_rotation,
            hadamard=self.hadamard + other.hadamard,
            s_gate=self.s_gate + other.s_gate,
            cnot=self.cnot + other.cnot,
            cz=self.cz + other.cz,
            and_gate=self.and_gate + other.and_gate,
            and_dagger_gate=self.and_dagger_gate + other.and_dagger_gate,
            bloqs_with_unknown_cost=frozendict(merged_unknowns),
        )

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def __mul__(self, other: SymbolicInt):
        if not isinstance(other, (int, SymbolicInt, sympy.Expr)):
            raise TypeError(
                f"Can only multiply by int, SymbolicInt or sympy Expr, not {type(other)}: {other}"
            )

        multiplied_unknowns = {
            bloq: count * other for bloq, count in self.bloqs_with_unknown_cost.items()
        }
        return FLASQGateCounts(
            t=other * self.t,
            toffoli=other * self.toffoli,
            z_rotation=other * self.z_rotation,
            x_rotation=other * self.x_rotation,
            hadamard=other * self.hadamard,
            s_gate=other * self.s_gate,
            cnot=other * self.cnot,
            cz=other * self.cz,
            and_gate=other * self.and_gate,
            and_dagger_gate=other * self.and_dagger_gate,
            bloqs_with_unknown_cost=frozendict(multiplied_unknowns),
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __str__(self):
        items_dict = self.asdict()
        if "bloqs_with_unknown_cost" in items_dict:
            unknown_dict = items_dict["bloqs_with_unknown_cost"]
            if isinstance(unknown_dict, Mapping):
                # Ensure consistent string representation for dict field
                unknown_str = (
                    "{"
                    + ", ".join(
                        f"{k!s}: {v!s}"
                        for k, v in sorted(unknown_dict.items(), key=lambda item: str(item[0]))
                    )
                    + "}"
                )
                items_dict["bloqs_with_unknown_cost"] = unknown_str

        # Sort items by key for consistent output
        strs = [f"{k}: {v}" for k, v in sorted(items_dict.items())]
        if strs:
            return ", ".join(strs)
        return "-"

    def asdict(self) -> Dict[str, Union[SymbolicInt, Dict["Bloq", SymbolicInt]]]:
        # Filter out zero counts and empty dicts
        d = attrs.asdict(
            self, recurse=False, filter=lambda a, v: not is_zero(v) and v != frozendict()
        )
        if "bloqs_with_unknown_cost" in d and isinstance(d["bloqs_with_unknown_cost"], frozendict):
            d["bloqs_with_unknown_cost"] = dict(d["bloqs_with_unknown_cost"])
        return d

    @property
    def total_rotations(self) -> SymbolicFloat:
        """Returns the sum of x_rotation and z_rotation counts."""
        return self.x_rotation + self.z_rotation


def _is_identity_exponent(exp: SymbolicFloat) -> bool:
    """Check if exponent is 0.0 or 1.0 (Pauli/identity)."""
    return bool(np.any(np.abs(exp - np.asarray([0.0, 1.0])) < 1e-11))


def _is_sqrt_exponent(exp: SymbolicFloat) -> bool:
    """Check if exponent is ±0.5 (sqrt gate)."""
    return bool(np.any(np.abs(exp - np.asarray([0.5, -0.5])) < 1e-11))


_ZERO_COST_TYPES = (XGate, YGate, ZGate, GlobalPhase, Identity)
_SIMPLE_GATE_MAP: dict[type, FLASQGateCounts] = {
    Hadamard: FLASQGateCounts(hadamard=1),
    SGate: FLASQGateCounts(s_gate=1),
    Toffoli: FLASQGateCounts(toffoli=1),
    CNOT: FLASQGateCounts(cnot=1),
    CZ: FLASQGateCounts(cz=1),
}


@frozen(kw_only=True)
class FLASQGateTotals(CostKey[FLASQGateCounts]):
    """Qualtran CostKey that recursively decomposes a circuit and produces FLASQGateCounts.

    Handles primitive gates (T, Toffoli, Rz, Rx, H, S, CNOT, CZ, And) directly.
    Pauli gates, measurements, states/effects, and bookkeeping bloqs are free.
    All other bloqs are decomposed recursively; those that cannot be decomposed
    or matched are recorded in ``bloqs_with_unknown_cost``.
    """

    def compute(
        self, bloq: "Bloq", get_callee_cost: Callable[["Bloq"], FLASQGateCounts]
    ) -> FLASQGateCounts:
        # Note: The execution order here is slightly optimized relative to the
        # list in Phase5.md (handling GlobalPhase, Identity, and certain
        # CirqGateAsBloq cases earlier) but is behaviorally equivalent.

        # 1. Zero-cost types
        if isinstance(bloq, _ZERO_COST_TYPES):
            return self.zero()

        # 2. PowGate identity
        if isinstance(bloq, (XPowGate, YPowGate, ZPowGate)):
            if _is_identity_exponent(bloq.exponent):
                return self.zero()

        # 3. T gates
        if bloq_is_t_like(bloq):
            return FLASQGateCounts(t=1)

        # 4. Simple gate lookup
        gate_type = type(bloq)
        if gate_type in _SIMPLE_GATE_MAP:
            return _SIMPLE_GATE_MAP[gate_type]

        # 5. And gates
        if isinstance(bloq, And):
            return FLASQGateCounts(
                and_dagger_gate=1 if bloq.uncompute else 0, and_gate=0 if bloq.uncompute else 1
            )

        # 6. CirqGateAsBloq special cases
        if isinstance(bloq, CirqGateAsBloq):
            if isinstance(bloq.cirq_gate, cirq.ZZPowGate):
                return FLASQGateCounts(cnot=2, z_rotation=1)
            if isinstance(bloq.cirq_gate, (cirq.MeasurementGate, cirq.ResetChannel)):
                return FLASQGateCounts()

        # 7. States/effects are free
        if bloq_is_state_or_effect(bloq):
            # Note: MeasureX, PrepX, MeasureZ, PrepZ are free in the surface code.
            # The FLASQ paper (Appendix C.10) bounds Y-basis operation costs at 0.5 blocks
            # (optimistic) or 1 block (conservative), citing Gidney, C. Quantum 8, 1310 (2024).
            # Decomposing Y-basis operations into an S gate and a Z/X measurement costs at least 1 block.
            # This exceeds the paper's bound, where costs coincide at 0.5 blocks for 'in motion' S gates.
            # Since Qualtran lacks specific Y-basis Bloqs and our examples do not use them, we defer implementation.
            # If added later, handle them explicitly here to use the lower paper bounds instead of decomposition.
            return FLASQGateCounts()

        # 8. Bookkeeping
        if isinstance(bloq, _BookkeepingBloq):
            return FLASQGateCounts()

        # 9. Explicit measurements
        if isinstance(bloq, (MeasureX, MeasureZ)):
            return FLASQGateCounts()
        # See note above about MeasureY/PrepY in step 7.

        # 10. PowGate sqrt decompositions
        if isinstance(bloq, ZPowGate) and _is_sqrt_exponent(bloq.exponent):
            return FLASQGateCounts(s_gate=1)
        if isinstance(bloq, XPowGate) and _is_sqrt_exponent(bloq.exponent):
            return FLASQGateCounts(hadamard=2, s_gate=1)
        if isinstance(bloq, YPowGate) and _is_sqrt_exponent(bloq.exponent):
            return FLASQGateCounts(hadamard=1)

        # 11. Unknown Cliffords
        if bloq_is_clifford(bloq):
            return FLASQGateCounts(bloqs_with_unknown_cost={bloq: 1})

        # 12. Arbitrary rotations
        if bloq_is_rotation(bloq):
            if isinstance(bloq, (Rz, ZPowGate)):
                return FLASQGateCounts(z_rotation=1)
            elif isinstance(bloq, (Rx, XPowGate)):
                return FLASQGateCounts(x_rotation=1)

        # 13. Recursive decomposition fallback
        callees = get_bloq_callee_counts(bloq, ignore_decomp_failure=True)
        if not callees:
            logger.debug("No decomposition or base case for FLASQ counts: %s", bloq)
            return FLASQGateCounts(bloqs_with_unknown_cost={bloq: 1})

        totals = self.zero()
        logger.info("Computing %s for %s from %d callee(s)", self, bloq, len(callees))
        for callee, n_times_called in callees:
            callee_cost = get_callee_cost(callee)
            totals += n_times_called * callee_cost
        return totals

    def zero(self) -> FLASQGateCounts:
        return FLASQGateCounts()

    def validate_val(self, val: FLASQGateCounts):
        if not isinstance(val, FLASQGateCounts):
            raise TypeError(f"{self} values should be `FLASQGateCounts`, got {type(val)}: {val}")

    def __str__(self):
        return "FLASQ gate totals"
