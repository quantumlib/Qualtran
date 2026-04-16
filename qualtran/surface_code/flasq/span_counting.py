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

"""Distance-dependent gate cost computation for the FLASQ model.

Computes the Manhattan distance (or rectilinear Steiner tree distance for
3+ qubits) between qubit operands to determine routing and interaction
costs. These distances are later scaled by connect_span_volume and
compute_span_volume in FLASQCostModel to produce ancilla volumes.
"""

import logging
from typing import Callable, Dict, Mapping, Sequence, Tuple, Union

import attrs
import sympy
from attrs import frozen
from frozendict import frozendict

from qualtran import Bloq, Signature
from qualtran.bloqs.basic_gates import Swap
from qualtran.bloqs.basic_gates.global_phase import GlobalPhase
from qualtran.bloqs.basic_gates.identity import Identity
from qualtran.bloqs.basic_gates.x_basis import MeasureX
from qualtran.bloqs.basic_gates.z_basis import MeasureZ
from qualtran.bloqs.bookkeeping._bookkeeping_bloq import _BookkeepingBloq
from qualtran.bloqs.mcmt import And
from qualtran.resource_counting import CostKey, get_bloq_callee_counts
from qualtran.surface_code.flasq.utils import _to_frozendict
from qualtran.symbolics import is_zero
from qualtran.symbolics.types import SymbolicFloat, SymbolicInt

logger = logging.getLogger(__name__)


@frozen(kw_only=True)
class BloqWithSpanInfo(Bloq):
    """An extension of Bloq that has information about the span of the Bloq."""

    wrapped_bloq: Bloq
    connect_span: SymbolicFloat  # The calculated span for the wrapped_bloq based on qubit layout.
    compute_span: SymbolicFloat  # The calculated span for the wrapped_bloq based on qubit layout.

    @property
    def signature(self) -> Signature:
        return self.wrapped_bloq.signature

    def build_composite_bloq(self, bb, **soqs):
        """Decomposes this bloq by adding the wrapped_bloq to the BloqBuilder."""
        return bb.add_d(self.wrapped_bloq, **soqs)

    def __str__(self):
        return f"BloqWithSpanInfo({self.wrapped_bloq}, connect_span={self.connect_span}, compute_span={self.compute_span})"

    def my_static_costs(self, cost_key):
        """Provide the stored span for TotalSpanCost."""
        if isinstance(cost_key, TotalSpanCost):
            return GateSpan(connect_span=self.connect_span, compute_span=self.compute_span)

        return NotImplemented


@frozen(kw_only=True)
class GateSpan:
    """Accumulated distance-dependent costs for multi-qubit gates.

    The paper uses p(q₁,...,qₖ) for the spanning distance between operands.

    Attributes:
        connect_span: Total Manhattan distance of corridors opened/closed for
            routing qubits. Corresponds to routing volume — the cost of
            moving occupied data qubits via walking surface codes.
        compute_span: Total distance over which lattice surgery merge/split
            is performed. Corresponds to base interaction volume.
        uncounted_bloqs: Multi-qubit bloqs lacking span information.

    Total distance-dependent ancilla volume =
        connect_span × connect_span_volume +
        compute_span × compute_span_volume.
    """

    connect_span: SymbolicFloat = 0
    compute_span: SymbolicFloat = 0
    uncounted_bloqs: Mapping[Bloq, SymbolicInt] = attrs.field(
        converter=_to_frozendict, default=frozendict()
    )

    def __add__(self, other):
        if not isinstance(other, GateSpan):
            if isinstance(other, int) and other == 0:
                return self
            raise TypeError(
                f"Can only add other `GateSpan` objects or 0, not {type(other)}: {other}"
            )
        # Merge frozendicts by summing counts for overlapping keys
        merged_uncounted = dict(self.uncounted_bloqs)
        for bloq, count in other.uncounted_bloqs.items():
            merged_uncounted[bloq] = merged_uncounted.get(bloq, 0) + count

        return GateSpan(
            connect_span=self.connect_span + other.connect_span,
            compute_span=self.compute_span + other.compute_span,
            uncounted_bloqs=frozendict(merged_uncounted),
        )

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def __mul__(self, other):
        if not isinstance(other, (int, float, sympy.Expr)):
            raise TypeError(
                f"Can only multiply by int, SymbolicInt or sympy Expr, not {type(other)}: {other}"
            )

        multiplied_uncounted = {bloq: count * other for bloq, count in self.uncounted_bloqs.items()}

        new_connect_span = self.connect_span * other
        new_compute_span = self.compute_span * other
        return GateSpan(
            connect_span=new_connect_span,
            compute_span=new_compute_span,
            uncounted_bloqs=frozendict(multiplied_uncounted),
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __str__(self):
        parts = []
        if not is_zero(self.connect_span):  # type: ignore[arg-type]
            parts.append(f"connect_span: {self.connect_span}")
        if not is_zero(self.compute_span):  # type: ignore[arg-type]
            parts.append(f"compute_span: {self.compute_span}")
        if self.uncounted_bloqs:
            uncounted_str = (
                "{"
                + ", ".join(
                    f"{k!s}: {v!s}"
                    for k, v in sorted(self.uncounted_bloqs.items(), key=lambda item: str(item[0]))
                )
                + "}"
            )
            parts.append(f"uncounted_bloqs: {uncounted_str}")
        if not parts:
            return "-"
        return ", ".join(parts)

    def asdict(self) -> Dict[str, Union[SymbolicInt, Dict["Bloq", SymbolicInt]]]:
        # Filter out zero counts and empty dicts
        d = attrs.asdict(
            self, recurse=False, filter=lambda a, v: not is_zero(v) and v != frozendict()
        )
        if "uncounted_bloqs" in d and isinstance(d["uncounted_bloqs"], frozendict):
            d["uncounted_bloqs"] = dict(d["uncounted_bloqs"])
        return d


def _calculate_spanning_distance(coords: Sequence[Tuple[int, ...]]) -> SymbolicInt:
    """Calculates the rectilinear spanning distance for a set of coordinates.

    - 2 qubits: Manhattan distance.
    - 3 qubits: Rectilinear Steiner tree distance.
    - >3 qubits: Not implemented.
    """
    n_qubits = len(coords)
    if n_qubits <= 1:
        return 0

    if n_qubits == 2:
        return abs(coords[0][0] - coords[1][0]) + abs(coords[0][1] - coords[1][1])

    if n_qubits == 3:
        d01 = abs(coords[0][0] - coords[1][0]) + abs(coords[0][1] - coords[1][1])
        d12 = abs(coords[1][0] - coords[2][0]) + abs(coords[1][1] - coords[2][1])
        d20 = abs(coords[2][0] - coords[0][0]) + abs(coords[2][1] - coords[0][1])
        total_distance = d01 + d12 + d20
        # The rectilinear Steiner distance for 3 points is half the perimeter of the bounding box.
        # This should always be an integer.
        assert total_distance % 2 == 0
        return total_distance // 2

    raise NotImplementedError(
        f"Spanning distance calculation for {n_qubits} qubits (> 3) is not implemented."
    )


def calculate_spans(
    coords: Sequence[Tuple[int, ...]], bloq: Bloq
) -> Tuple[SymbolicFloat, SymbolicFloat]:
    """Calculates connect_span and compute_span for a bloq given qubit coordinates.

    This function centralizes the logic for determining span costs. It first
    calculates a base `spanning_distance` and then applies bloq-specific rules
    to derive `connect_span` and `compute_span`.

    Args:
        coords: A list of (row, col) tuples for each qubit the bloq acts on.
        bloq: The bloq being costed.

    Returns:
        A tuple of (connect_span, compute_span).
    """
    if len(coords) != bloq.signature.n_qubits():
        raise ValueError(
            f"Number of coordinates ({len(coords)}) does not match number of "
            f"qubits for bloq {bloq} ({bloq.signature.n_qubits()})."
        )

    if bloq.signature.n_qubits() <= 1:
        return 0, 0

    # Apply bloq-specific rules
    if isinstance(bloq, And) and bloq.uncompute:
        # For And(uncompute=True), distance is between the two control qubits.
        # Cost is like CZ (D, D). (Appendix A.1.10)
        spanning_distance = _calculate_spanning_distance(coords[:2])
        return spanning_distance, spanning_distance

    spanning_distance = _calculate_spanning_distance(coords)
    # Handle Swap gate explicitly (Appendix A.1.9 implies connect=D, compute=2D)
    if isinstance(bloq, Swap):
        connect_span = spanning_distance
        compute_span = 2 * spanning_distance
        return connect_span, compute_span

    # Default rule for other multi-qubit gates (CNOT, CZ, And, Toffoli, Move)
    # Updated to (D, D) based on Appendix A.1.
    connect_span = spanning_distance
    compute_span = spanning_distance
    return connect_span, compute_span


def bloq_is_not_multiqubit(bloq: Bloq):
    """Checks if the given bloq acts on at most one qubit."""
    return bloq.signature.n_qubits() <= 1


@frozen(kw_only=True)
class TotalSpanCost(CostKey[GateSpan]):
    """Qualtran CostKey that computes distance-dependent gate costs via qubit placement geometry.

    Recursively traverses a bloq decomposition tree and sums GateSpan costs.
    Single-qubit bloqs and bookkeeping bloqs have zero span. Multi-qubit bloqs
    without span info or decomposition are recorded in ``uncounted_bloqs``.
    """

    def compute(self, bloq: Bloq, get_callee_cost: Callable[[Bloq], GateSpan]) -> GateSpan:

        # Base case: Span is zero for single-qubit bloqs or specific types
        if bloq_is_not_multiqubit(bloq):
            logger.debug("Zero span (single qubit) for %s", bloq)
            return self.zero()
        if isinstance(bloq, (MeasureX, MeasureZ, GlobalPhase, Identity, _BookkeepingBloq)):
            logger.debug("Zero span (special type) for %s", bloq)
            return self.zero()

        # Recursive case: sum up the spans of the callees.
        callees = get_bloq_callee_counts(bloq, ignore_decomp_failure=True)

        if not callees:
            # If no decomposition and not handled above, mark as uncounted.
            logger.debug("No decomposition or static cost for multi-qubit bloq %s", bloq)
            return GateSpan(uncounted_bloqs={bloq: 1})

        # Decompose and sum costs recursively.
        logger.info("Computing %s for %s from %d callee(s)", self, bloq, len(callees))
        total_cost = self.zero()
        for callee, n_times_called in callees:
            callee_cost = get_callee_cost(callee)
            total_cost += n_times_called * callee_cost
        return total_cost

    def zero(self) -> GateSpan:
        return GateSpan(connect_span=0, compute_span=0)

    def validate_val(self, val: GateSpan):
        if not isinstance(val, GateSpan):
            raise TypeError(f"{self} values should be `GateSpan`, got {type(val)}: {val}")

    def __str__(self):
        return "total span cost"
