"""Measurement depth computation for the FLASQ cost model.

Upper-bounds the sequential measurement chain length (the paper's
'measurement depth bound') by finding the longest path in the circuit
DAG, where each node is weighted by its measurement depth contribution.
Also called 'reaction depth' in the paper.
"""

import attrs
from attrs import frozen
from frozendict import frozendict
import sympy
from typing import Dict, Any, Union, Callable, Optional, Mapping
import logging
import networkx as nx

# Qualtran Imports
from qualtran import (
    Bloq,
    CompositeBloq,
    DanglingT,
    DecomposeNotImplementedError,
    DecomposeTypeError,
    BloqInstance,
)
from qualtran.bloqs.basic_gates import (
    TGate,
    Toffoli,
    GlobalPhase,
    Identity,
    Rx,
    XPowGate,
    Rz,
    ZPowGate,
)
from qualtran.bloqs.bookkeeping._bookkeeping_bloq import _BookkeepingBloq
from qualtran.bloqs.mcmt import And
from qualtran.resource_counting import CostKey
from qualtran.resource_counting.classify_bloqs import (
    bloq_is_state_or_effect,
    bloq_is_clifford,
)
from qualtran.symbolics import is_zero, SymbolicFloat, SymbolicInt

logger = logging.getLogger(__name__)


@frozen(kw_only=True)
class MeasurementDepth:
    """Represents measurement depth and tracks bloqs with unknown depth.

    This class is immutable and hashable.

    Attributes:
        depth: The calculated measurement depth, potentially symbolic.
        bloqs_with_unknown_depth: An immutable mapping from bloqs whose measurement
            depth is unknown to the count of such bloqs encountered.
    """

    depth: SymbolicFloat = 0
    # Use frozendict for immutability and hashability.
    # Input can be a dict, but it's converted in __attrs_post_init__.
    bloqs_with_unknown_depth: Mapping[Bloq, SymbolicInt] = attrs.field(
        factory=frozendict
    )

    def __attrs_post_init__(self):
        """Ensure bloqs_with_unknown_depth is always a frozendict."""
        if not isinstance(self.bloqs_with_unknown_depth, frozendict):
            # Use object.__setattr__ because the class is frozen
            object.__setattr__(
                self,
                "bloqs_with_unknown_depth",
                frozendict(self.bloqs_with_unknown_depth),
            )

    def __add__(self, other: "MeasurementDepth") -> "MeasurementDepth":
        """Adds two MeasurementDepth objects.

        Depths are summed, and unknown bloq counts are merged by summing.
        """
        if not isinstance(other, MeasurementDepth):
            # Allow adding zero (identity)
            if isinstance(other, int) and other == 0:
                return self
            return NotImplemented

        new_depth = self.depth + other.depth

        # Merge frozendicts by summing counts for overlapping keys
        merged_unknowns = dict(self.bloqs_with_unknown_depth)
        for bloq, count in other.bloqs_with_unknown_depth.items():
            merged_unknowns[bloq] = merged_unknowns.get(bloq, 0) + count

        # The constructor handles converting the merged dict back to frozendict
        return MeasurementDepth(
            depth=new_depth, bloqs_with_unknown_depth=merged_unknowns
        )

    def __radd__(self, other):
        """Handles reversed addition, e.g., sum([MeasurementDepth(...)])"""
        if other == 0:
            return self
        return self.__add__(other)

    def __str__(self) -> str:
        """Returns a human-readable string representation."""
        items = self.asdict()
        if not items:
            return "MeasurementDepth(depth: 0)"

        str_items = []
        if "depth" in items:
            str_items.append(f"depth: {items['depth']}")
        if "bloqs_with_unknown_depth" in items:
            unknown_dict = items["bloqs_with_unknown_depth"]
            try:
                # Sort unknown bloqs by string representation for consistent output
                sorted_unknown = sorted(
                    unknown_dict.items(), key=lambda item: str(item[0])
                )
            except TypeError:  # pragma: no cover
                # Fallback if keys somehow aren't comparable via string
                sorted_unknown = unknown_dict.items()
            unknown_str = (
                "{" + ", ".join(f"{k!s}: {v!s}" for k, v in sorted_unknown) + "}"
            )
            str_items.append(f"bloqs_with_unknown_depth: {unknown_str}")

        return f"MeasurementDepth({', '.join(sorted(str_items))})"

    def asdict(self) -> Dict[str, Union[SymbolicFloat, Mapping[Bloq, SymbolicInt]]]:
        """Returns a dictionary representation, filtering zero depth and empty unknowns."""
        # Use attrs.asdict, filtering out fields that are zero/empty
        d = attrs.asdict(
            self,
            recurse=False,
            filter=lambda attr, value: not (
                (attr.name == "depth" and is_zero(value))
                or (attr.name == "bloqs_with_unknown_depth" and not value)
            ),
        )
        return d


def _cbloq_measurement_depth(
    cbloq: CompositeBloq, get_callee_cost: Callable[[Bloq], MeasurementDepth]
) -> MeasurementDepth:
    """Calculates the measurement depth of a CompositeBloq using the longest path.

    This function traverses the directed acyclic graph (DAG) representing the
    CompositeBloq. It assigns the measurement depth of each sub-bloq (obtained via
    `get_callee_cost`) as a weight to the *outgoing* edges from that sub-bloq's
    node in the graph. It then computes the longest path through this weighted DAG.
    Any sub-bloqs encountered for which the cost cannot be determined are collected.

    Args:
        cbloq: The CompositeBloq to analyze. Must be a DAG.
        get_callee_cost: A function that returns the MeasurementDepth cost
        (depth value + unknown bloqs) for a given sub-bloq.

    Returns:
        A MeasurementDepth object representing the total depth (longest path length)
        and a merged mapping of any unknown bloqs encountered during the traversal.
    """
    binst_graph = cbloq._binst_graph.copy()
    # Use a mutable dict for efficient aggregation during the loop
    total_unknown_bloqs_mut: Dict[Bloq, SymbolicInt] = {}

    # 1. Assign weights to edges based on the source node's measurement depth
    for node in binst_graph.nodes():
        node_depth: SymbolicFloat = 0
        node_unknowns: Mapping[Bloq, SymbolicInt] = frozendict()

        if isinstance(node, BloqInstance):
            bloq = node.bloq
            try:
                # Recursively get the cost of the sub-bloq
                node_cost = get_callee_cost(bloq)
                node_depth = node_cost.depth
                node_unknowns = node_cost.bloqs_with_unknown_depth
            except Exception as e:
                # If cost calculation fails for a sub-bloq, log error,
                # assign 0 depth, and mark the sub-bloq as unknown.
                logger.error(
                    f"Error getting measurement depth cost for sub-bloq {bloq} "
                    f"within {cbloq}: {e}",
                    exc_info=True,
                )
                node_depth = 0
                node_unknowns = frozendict({bloq: 1})

            # Merge unknown bloqs from this node into the total
            for unknown_b, count in node_unknowns.items():
                total_unknown_bloqs_mut[unknown_b] = (
                    total_unknown_bloqs_mut.get(unknown_b, 0) + count
                )

        elif isinstance(node, DanglingT):
            # Dangling edges represent graph inputs/outputs, contribute 0 depth.
            node_depth = 0
        else:  # pragma: no cover
            # This case should ideally not be reached with valid CompositeBloqs
            logger.warning(
                f"Unexpected node type {type(node)} found in binst_graph for {cbloq}. "
                "Assigning 0 depth."
            )
            node_depth = 0

        # Assign the depth of the current node (source of the edge)
        # as a weight to all its outgoing edges.
        for u, v in binst_graph.out_edges(node):
            # Hardcode the weight key string directly
            binst_graph.edges[u, v]["measurement_depth_from_tail"] = node_depth

    # Calculate the longest path length using the assigned edge weights.
    # `dag_longest_path_length` sums the weights *along* the path.
    longest_path_length = nx.dag_longest_path_length(
        binst_graph, weight="measurement_depth_from_tail"
    )

    return MeasurementDepth(
        depth=longest_path_length,
        bloqs_with_unknown_depth=frozendict(total_unknown_bloqs_mut),
    )


@frozen(kw_only=True)
class TotalMeasurementDepth(CostKey[MeasurementDepth]):
    """Qualtran CostKey computing an upper bound on measurement depth.

    The paper calls this the "Measurement Depth Bound". Determines the
    longest path in a computation DAG where each edge is weighted by the
    measurement depth of the operation at its source node. In the current
    implementation, only T/Toffoli/And gates contribute non-zero
    measurement depth.

    Assumed Base Depths:
     - Clifford gates (including CNOT, Hadamard, X, Z, S), state preparations,
       measurements, Identity, and GlobalPhase have a depth of 0.
     - T gates, Toffoli gates, and And gates have a depth of 1.

    Bloqs without a defined decomposition or base case contribute 0 to the depth
    but are recorded in the `bloqs_with_unknown_depth` field of the result.

    The cost value type for this CostKey is `MeasurementDepth`.

    Attributes:
        rotation_depth: When set, assigns this value as the measurement
            depth for rotation bloqs (Rx, Rz, XPowGate, ZPowGate). This
            allows deferring the rotation depth calculation until the
            per-rotation error budget is determined. When None, rotation
            bloqs are flagged as unknown.
    """

    rotation_depth: Optional[float] = None

    def compute(
        self, bloq: Bloq, get_callee_cost: Callable[[Bloq], MeasurementDepth]
    ) -> MeasurementDepth:
        """Compute the measurement depth for the given bloq."""

        # --- Base Cases ---
        if isinstance(bloq, (And, TGate, Toffoli)):
            return MeasurementDepth(depth=1)

        # Bookkeeping and Phase gates have depth 0
        if isinstance(bloq, (Identity, _BookkeepingBloq, GlobalPhase)):
            return MeasurementDepth(depth=0)

        # Clifford gates and State Preparation/Measurement have depth 0
        if isinstance(bloq, Bloq):  # Ensure it's a Bloq before checking properties
            if bloq_is_clifford(bloq):
                return MeasurementDepth(depth=0)
            if bloq_is_state_or_effect(bloq):
                return MeasurementDepth(depth=0)

        # We might assign a fixed cost for rotation bloqs.
        if self.rotation_depth is not None:
            if isinstance(bloq, (Rx, Rz, XPowGate, ZPowGate)):
                return MeasurementDepth(depth=self.rotation_depth)

        # --- Recursive Case ---
        cbloq: Optional[CompositeBloq] = None
        if isinstance(bloq, CompositeBloq):
            # If it's already a CompositeBloq, analyze its graph directly
            logger.debug(
                "Computing %s using provided CompositeBloq graph for %s", self, bloq
            )
            cbloq = bloq
        else:
            # Otherwise, try to decompose the bloq
            try:
                cbloq = bloq.decompose_bloq()
                logger.debug("Computing %s for %s from its decomposition", self, bloq)
            except (DecomposeNotImplementedError, DecomposeTypeError):
                # Decomposition failed or not implemented, proceed to fallback
                logger.debug("Decomposition failed for %s, using fallback.", bloq)
            except Exception as e:  # pragma: no cover
                # Catch unexpected errors during decomposition itself
                logger.error(
                    f"Unexpected error during decomposition of {bloq}: {e}",
                    exc_info=True,
                )

        if isinstance(cbloq, CompositeBloq):
            return _cbloq_measurement_depth(cbloq, get_callee_cost)

        # --- Fallback Case ---
        # If no base case matched and decomposition was not possible/successful,
        # mark the bloq as having unknown measurement depth.
        logger.debug(
            "No base case or decomposition found for %s regarding measurement depth. "
            "Marking as unknown.",
            bloq,
        )
        return MeasurementDepth(depth=0, bloqs_with_unknown_depth=frozendict({bloq: 1}))

    def zero(self) -> MeasurementDepth:
        """Returns the additive identity for MeasurementDepth (zero depth, no unknowns)."""
        # The MeasurementDepth factory defaults to zero depth and empty frozendict
        return MeasurementDepth()

    def validate_val(self, val: MeasurementDepth):
        """Validates that the computed cost value is an instance of MeasurementDepth."""
        if not isinstance(val, MeasurementDepth):
            raise TypeError(
                f"{self} cost values must be `MeasurementDepth`, got {type(val)}: {val}"
            )

    def __str__(self):
        """Return a descriptive string for this cost key."""
        return "total measurement depth"
