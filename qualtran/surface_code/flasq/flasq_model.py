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

"""Core FLASQ cost model for fault-tolerant quantum resource estimation.

Defines FLASQCostModel (gate volume parameters for conservative and optimistic
models), FLASQSummary (computed resource estimates including spacetime volume,
depth, and qubit counts), and the main analysis function apply_flasq_cost_model.
"""

import logging
import typing
import warnings
from functools import lru_cache
from typing import Mapping, Optional

import sympy
from attrs import field, fields, frozen
from frozendict import frozendict

from qualtran.surface_code.flasq.measurement_depth import MeasurementDepth
from qualtran.surface_code.flasq.span_counting import GateSpan
from qualtran.surface_code.flasq.symbols import (
    MIXED_FALLBACK_T_COUNT,
    ROTATION_ERROR,
    T_REACT,
    V_CULT_FACTOR,
)
from qualtran.surface_code.flasq.utils import substitute_until_fixed_point
from qualtran.surface_code.flasq.volume_counting import FLASQGateCounts
from qualtran.symbolics import SymbolicFloat, SymbolicInt

# Initialize logger
logger = logging.getLogger(__name__)


@frozen(kw_only=True)
class FLASQCostModel:
    """A cost model assigning ancilla spacetime volumes to primitive FLASQ gates.

    All ``_volume`` attributes are **ancilla** spacetime volumes, not total
    spacetime volumes (which also include idling). The paper is explicit
    about this distinction.

    Base costs default to conservative values. Derived costs for Toffoli,
    And, and rotation gates are calculated from these base costs if not
    explicitly provided.

    Attributes:
        t_clifford_volume: Lattice surgery overhead for a T gate excluding
            cultivation. Equals Vol(T) − Vol(Cultivate) in the paper.
        t_cultivation_volume: Cultivation ancilla volume per T gate.
        h_volume: Ancilla volume for Hadamard gate.
        s_volume: Ancilla volume for S gate.
        cnot_base_volume: Base ancilla volume for CNOT (0 in both models;
            full cost comes from span).
        cz_base_volume: Base ancilla volume for CZ (0 in both models;
            full cost comes from span).
        connect_span_volume: Per-unit-distance cost for opening/closing a
            routing corridor via walking surface codes.
            Conservative=4, Optimistic=1.
        compute_span_volume: Per-unit-distance cost for the actual lattice
            surgery merge/split. Both models=1. Combined with connect_span:
            total distance-dependent ancilla vol =
            connect_span × connect_span_volume +
            compute_span × compute_span_volume.
        toffoli_clifford_volume: Lattice surgery overhead for Toffoli gate.
        and_clifford_volume: Lattice surgery overhead for And gate.
        and_dagger_clifford_volume: Lattice surgery overhead for And† gate.
        toffoli_cultivation_volume: Cultivation volume for Toffoli (derived).
        and_cultivation_volume: Cultivation volume for And gate (derived).
        rz_clifford_volume: Lattice surgery overhead for Rz rotation (derived).
        rz_cultivation_volume: Cultivation volume for Rz rotation (derived).
        rx_clifford_volume: Lattice surgery overhead for Rx rotation (derived).
        rx_cultivation_volume: Cultivation volume for Rx rotation (derived).
    """

    # --- Base Parameters with Conservative Defaults ---
    t_clifford_volume: SymbolicFloat = T_REACT + 6.0
    t_cultivation_volume: SymbolicFloat = 1.5 * V_CULT_FACTOR
    h_volume: SymbolicFloat = 7.0
    s_volume: SymbolicFloat = 5.5
    cnot_base_volume: SymbolicFloat = 0.0
    cz_base_volume: SymbolicFloat = 0.0
    connect_span_volume: SymbolicFloat = 4.0
    compute_span_volume: SymbolicFloat = 1.0

    toffoli_clifford_volume: SymbolicFloat = 5 * T_REACT + 68.0
    and_clifford_volume: SymbolicFloat = 2 * T_REACT + 64.0
    and_dagger_clifford_volume: SymbolicFloat = 0.0

    # --- New Parameters for Rotation Synthesis Overhead ---

    # Conservative Defaults (d=0, App A.1.7):
    # Ancilla holding time per T-gate during rotation synthesis.
    extra_cost_per_t_gate_in_rotation: SymbolicFloat = 2.0
    # Fixed overhead per rotation (Clifford gates, CNOTs, constant terms).
    # Calculation: (2*Vs + 2*Vh + 2*Vcnot_adj + 10) = (11 + 14 + 10 + 10) = 45.0
    extra_cost_per_rotation: SymbolicFloat = 45.0

    # --- Derived Parameters (calculated in post_init if not provided) ---
    toffoli_cultivation_volume: Optional[SymbolicFloat] = field(default=None)
    and_cultivation_volume: Optional[SymbolicFloat] = field(default=None)
    rz_clifford_volume: Optional[SymbolicFloat] = field(default=None)
    rz_cultivation_volume: Optional[SymbolicFloat] = field(default=None)
    rx_clifford_volume: Optional[SymbolicFloat] = field(default=None)
    rx_cultivation_volume: Optional[SymbolicFloat] = field(default=None)

    def __attrs_post_init__(self):
        """Calculate and set default values for derived cost parameters."""
        # Use object.__setattr__ to modify fields in a frozen attrs class
        if self.toffoli_cultivation_volume is None:
            object.__setattr__(self, "toffoli_cultivation_volume", 4 * self.t_cultivation_volume)
        if self.and_cultivation_volume is None:
            object.__setattr__(self, "and_cultivation_volume", 4 * self.t_cultivation_volume)

        # Rotation costs (Simplified approach based on user request)

        t = MIXED_FALLBACK_T_COUNT  # The T-count formula

        # 1. Cultivation Volume
        rotation_cultivation_volume = t * self.t_cultivation_volume

        # 2. Clifford Volume (using new parameters)
        # Cost = t * (Vol(T_Clifford) + extra_cost_per_t_gate) + extra_cost_per_rotation
        rotation_clifford_volume = (
            t * (self.t_clifford_volume + self.extra_cost_per_t_gate_in_rotation)
            + self.extra_cost_per_rotation
        )

        if self.rz_cultivation_volume is None:
            object.__setattr__(self, "rz_cultivation_volume", rotation_cultivation_volume)
        if self.rx_cultivation_volume is None:
            object.__setattr__(self, "rx_cultivation_volume", rotation_cultivation_volume)
        if self.rz_clifford_volume is None:
            object.__setattr__(self, "rz_clifford_volume", rotation_clifford_volume)
        if self.rx_clifford_volume is None:
            object.__setattr__(self, "rx_clifford_volume", rotation_clifford_volume)

    # Data-driven mapping for volume calculations
    _PURE_CLIFFORD_VOLUME_MAP: Mapping[str, str] = frozendict(
        {
            "hadamard": "h_volume",
            "s_gate": "s_volume",
            "cnot": "cnot_base_volume",
            "cz": "cz_base_volume",
            "and_dagger_gate": "and_dagger_clifford_volume",
        }
    )

    _NON_CLIFFORD_LATTICE_SURGERY_VOLUME_MAP: Mapping[str, str] = frozendict(
        {
            "t": "t_clifford_volume",
            "toffoli": "toffoli_clifford_volume",
            "and_gate": "and_clifford_volume",
            "z_rotation": "rz_clifford_volume",
            "x_rotation": "rx_clifford_volume",
        }
    )

    _CULTIVATION_VOLUME_MAP: Mapping[str, str] = frozendict(
        {
            "t": "t_cultivation_volume",
            "toffoli": "toffoli_cultivation_volume",
            "and_gate": "and_cultivation_volume",
            "z_rotation": "rz_cultivation_volume",
            "x_rotation": "rx_cultivation_volume",
        }
    )

    def calculate_volume_required_for_clifford_computation(
        self, counts: FLASQGateCounts, span_info: GateSpan, verbose: bool = False
    ) -> SymbolicFloat:
        """Calculates the volume for pure Clifford gates and span-related overhead.


        Args:
            counts: The FLASQGateCounts object containing primitive gate counts.
            span_info: The GateSpan object containing total span.
            verbose: If True, print a detailed breakdown of volume contributions.

        Returns:
            The total calculated volume, potentially symbolic. Issues warnings if
            unknown/uncounted bloqs were present in the inputs.
        """
        if counts.bloqs_with_unknown_cost:
            warnings.warn(
                f"Calculating pure Clifford volume with unknown FLASQ counts: {counts.bloqs_with_unknown_cost}. "
                "Clifford volume result will be incomplete."
            )
        if span_info.uncounted_bloqs:
            warnings.warn(
                f"Calculating total Clifford volume with uncounted span bloqs: {span_info.uncounted_bloqs}. "
                "Clifford volume result will be incomplete."
            )

        if verbose:
            print("\n--- Clifford Volume Breakdown ---")

        total_volume: SymbolicFloat = 0
        for count_name, vol_name in self._PURE_CLIFFORD_VOLUME_MAP.items():
            count = getattr(counts, count_name)
            volume_per_gate = getattr(self, vol_name)
            term = count * volume_per_gate
            if term:
                if verbose:
                    # Create a more readable name for printing
                    print_name = count_name.replace("_", " ").title()
                    print(f"  {print_name} volume: {term}")
                total_volume += term

        # Add volume from connect_span
        term_connect_span = span_info.connect_span * self.connect_span_volume
        if term_connect_span:
            if verbose:
                print(f"  Connect Span volume: {term_connect_span}")
            total_volume += term_connect_span

        # Add volume from compute_span
        term_compute_span = span_info.compute_span * self.compute_span_volume
        if term_compute_span:
            if verbose:
                print(f"  Compute Span volume: {term_compute_span}")
            total_volume += term_compute_span

        return sympy.simplify(total_volume) if isinstance(total_volume, sympy.Expr) else total_volume

    def calculate_non_clifford_lattice_surgery_volume(
        self, counts: FLASQGateCounts, verbose: bool = False
    ) -> SymbolicFloat:
        """Calculates the lattice surgery volume for implementing non-Clifford gates.

        This captures the volume from the Clifford components of non-Clifford
        gates (e.g., the Clifford part of a T-gate implementation), but does
        not include cultivation costs.

        Args:
            counts: The FLASQGateCounts object containing primitive gate counts.
            verbose: If True, print a detailed breakdown of volume contributions.

        Returns:
            The total calculated volume, potentially symbolic.
        """
        if counts.bloqs_with_unknown_cost:
            warnings.warn(
                f"Calculating non-Clifford lattice surgery volume with unknown FLASQ counts: {counts.bloqs_with_unknown_cost}. "
                "Result will be incomplete."
            )

        if verbose:
            print("\n--- Non-Clifford Lattice Surgery Volume Breakdown ---")

        total_volume: SymbolicFloat = 0
        for count_name, vol_name in self._NON_CLIFFORD_LATTICE_SURGERY_VOLUME_MAP.items():
            count = getattr(counts, count_name)
            volume_per_gate = getattr(self, vol_name)
            term = count * volume_per_gate
            if term:
                if verbose:
                    print_name = count_name.replace("_", " ").title()
                    print(f"  {print_name} volume: {term}")
                total_volume += term

        return sympy.simplify(total_volume) if isinstance(total_volume, sympy.Expr) else total_volume

    def calculate_volume_required_for_cultivation(
        self, counts: FLASQGateCounts, verbose: bool = False
    ) -> SymbolicFloat:
        """Calculates the total volume required for T-state cultivation based on gate counts.

        Args:
            counts: The FLASQGateCounts object containing primitive gate counts.
            verbose: If True, print a detailed breakdown of volume contributions.

        Returns:
            The total calculated cultivation volume, potentially symbolic. Issues warnings if
            unknown counts were present.
        """
        if counts.bloqs_with_unknown_cost:
            warnings.warn(
                f"Calculating total cultivation volume with unknown FLASQ counts: {counts.bloqs_with_unknown_cost}. "
                "Cultivation volume result will be incomplete."
            )

        total_volume: SymbolicFloat = 0

        if verbose:
            print("\n--- Cultivation Volume Breakdown ---")

        # Loop through the gate counts and add their cultivation volumes
        for count_name, vol_name in self._CULTIVATION_VOLUME_MAP.items():
            count = getattr(counts, count_name)
            volume_per_gate = getattr(self, vol_name)
            term = count * volume_per_gate
            if term:
                if verbose:
                    print_name = count_name.replace("_", " ").title()
                    print(f"  {print_name} volume: {term}")
                total_volume += term

        return total_volume


# --- Standalone Calculation Functions ---


@frozen(kw_only=True)
class FLASQSummary:
    """Summarizes key FLASQ resource estimates including volumes and depth.

    Attributes:
        total_depth: L in the paper. Units = logical timesteps
            (each = d surface code cycles).
        n_algorithmic_qubits: Q (MaximumQubitUsage) in the paper.
        n_fluid_ancilla: A in the paper. Available fluid ancilla qubits.
        total_t_count: Total T gates, including those from Toffolis and
            rotations.
        total_rotation_count: Total Rx and Rz rotation gates.
        measurement_depth_val: Raw, unscaled measurement depth D.
        volume_limited_depth: V/A. Depth set by computational volume and
            available fluid ancillas.
        scaled_measurement_depth: D scaled by reaction time. Comparable
            to volume_limited_depth.
        total_computational_volume: V in the paper. Sum of ancilla volumes
            over all gates.
        clifford_computational_volume: Clifford component of V (H, S,
            CNOT, span overhead).
        non_clifford_lattice_surgery_volume: Non-Clifford lattice surgery
            component of V (T, Toffoli), excluding cultivation.
        cultivation_volume: Cultivation component of V.
        idling_volume: LQ term in S = LQ + V. Algorithmic qubits idling
            while gates execute on ancilla.
        total_clifford_volume: clifford_computational_volume + idling_volume.
            Includes idling because identity ∈ Clifford group.
        total_spacetime_volume: S = LQ + V in the paper.
        is_volume_limited: True when V/A ≥ t_react·D. Paper calls this
            "spacetime limited."
        is_reaction_limited: True when t_react·D > V/A.
    """

    total_clifford_volume: SymbolicFloat
    total_depth: SymbolicFloat
    n_algorithmic_qubits: SymbolicInt
    n_fluid_ancilla: SymbolicInt
    total_t_count: SymbolicFloat
    total_rotation_count: SymbolicFloat
    # New fields
    measurement_depth_val: SymbolicFloat
    volume_limited_depth: SymbolicFloat
    scaled_measurement_depth: SymbolicFloat
    total_computational_volume: SymbolicFloat
    idling_volume: SymbolicFloat
    clifford_computational_volume: SymbolicFloat
    non_clifford_lattice_surgery_volume: SymbolicFloat
    cultivation_volume: SymbolicFloat
    total_spacetime_volume: SymbolicFloat

    @property
    def is_volume_limited(self) -> bool:
        """Indicates if the computation's depth is limited by volume, not reaction time.

        This occurs if the total depth is determined by the
        spacetime volume required for its operations, rather than by sequential
        measurement dependencies (reaction time). This occurs when the
        volume-limited depth is greater than or equal to the measurement depth
        scaled by the reaction time.

        Raises:
            ValueError: If the summary contains unresolved symbols, preventing a
                boolean determination. Call `resolve_symbols` first.
        """
        comparison = self.volume_limited_depth >= self.scaled_measurement_depth
        try:
            return bool(comparison)
        except TypeError:
            raise ValueError(
                "Cannot determine if summary is volume-limited because it contains "
                "unresolved symbols. Call `resolve_symbols()` first."
            ) from None

    @property
    def is_reaction_limited(self) -> bool:
        """Indicates if the computation's depth is limited by reaction time, not volume.

        This occurs if the total depth is determined by
        the "reaction time" required for sequential measurements, rather than
        by the spacetime volume of its operations. This occurs when the
        measurement depth scaled by the reaction time is greater than the
        volume-limited depth.

        Raises:
            ValueError: If the summary contains unresolved symbols, preventing a
                boolean determination. Call `resolve_symbols` first.
        """
        comparison = self.scaled_measurement_depth > self.volume_limited_depth
        try:
            return bool(comparison)
        except TypeError:
            raise ValueError(
                "Cannot determine if summary is reaction-limited because it contains "
                "unresolved symbols. Call `resolve_symbols()` first."
            ) from None

    @property
    def regular_spacetime_volume(self) -> SymbolicFloat:
        """Spacetime volume excluding cultivation: S - V_cult.

        Used in error mitigation to compute the volume exposed to
        memory/logical errors (cultivation errors handled separately).
        """
        return self.total_spacetime_volume - self.cultivation_volume

    @lru_cache(maxsize=None)
    def resolve_symbols(
        self, assumptions: frozendict[typing.Union[sympy.Symbol, str], typing.Any]
    ) -> "FLASQSummary":
        """Substitutes symbols in the summary fields based on provided assumptions.

        Args:
            assumptions: A frozendict mapping sympy Symbols (or their string names)
                to the values they should be substituted with.

        Returns:
            A new FLASQSummary object with symbols resolved according to the assumptions.
            Fields that resolve to numbers will be converted to `int` or `float`.
        """
        resolved_fields = {}
        for field_to_process in fields(FLASQSummary):
            val = getattr(self, field_to_process.name)
            resolved_val = substitute_until_fixed_point(val, assumptions, try_make_number=True)
            resolved_fields[field_to_process.name] = resolved_val

        return FLASQSummary(**resolved_fields)


def apply_flasq_cost_model(
    model: FLASQCostModel,
    n_total_logical_qubits: SymbolicInt,
    qubit_counts: SymbolicInt,
    counts: FLASQGateCounts,
    span_info: GateSpan,
    measurement_depth: MeasurementDepth,
    logical_timesteps_per_measurement: SymbolicFloat,
    assumptions: Optional[frozendict] = None,
    verbosity: int = 0,
) -> FLASQSummary:
    """Calculates key FLASQ resource estimates including volumes and limiting depth.

    Args:
        model: The FLASQCostModel instance containing volume parameters.
        n_total_logical_qubits: The total number of logical qubits available.
        qubit_counts: Q (MaximumQubitUsage) in the paper. Number of
            algorithmic qubits (data + ancilla) used by the bloq.
        counts: The FLASQGateCounts object containing primitive gate counts.
        span_info: The GateSpan object containing total span.
        measurement_depth: The calculated measurement depth.
        logical_timesteps_per_measurement: Scales raw measurement depth by
            reaction time. Computed upstream as
            reaction_time_in_cycles / code_distance.
        assumptions: An optional frozendict of symbol substitutions to apply
            to the final summary. If provided, the returned summary will have
            its symbolic values resolved.
        verbosity: Controls print output level. 0=silent, 1=summary
            statistics, 2=detailed volume breakdowns.

    Returns:
        A FLASQSummary object containing the calculated total Clifford volume and total depth.
    """
    # Derive algorithmic and fluid qubit counts from the inputs.
    n_algorithmic_qubits = qubit_counts
    n_fluid_ancilla = n_total_logical_qubits - n_algorithmic_qubits

    # Issue warnings if inputs have unknowns/uncounted items
    if counts.bloqs_with_unknown_cost:
        warnings.warn(
            f"Calculating FLASQ summary with unknown FLASQ counts: {counts.bloqs_with_unknown_cost}. "
            "Result will be incomplete."
        )
    if span_info.uncounted_bloqs:
        warnings.warn(
            f"Calculating FLASQ summary with uncounted span bloqs: {span_info.uncounted_bloqs}. "
            "Result will be incomplete."
        )
    if measurement_depth.bloqs_with_unknown_depth:
        warnings.warn(
            f"Calculating FLASQ summary with unknown measurement depth bloqs: {measurement_depth.bloqs_with_unknown_depth}. "
            "Result will be incomplete."
        )

    # Calculate component volumes.
    # This is where we implement the plan to split the volume calculations.
    clifford_computational_vol = model.calculate_volume_required_for_clifford_computation(
        counts, span_info, verbose=(verbosity >= 2)
    )
    non_clifford_lattice_surgery_vol = model.calculate_non_clifford_lattice_surgery_volume(
        counts, verbose=(verbosity >= 2)
    )
    cultivation_volume = model.calculate_volume_required_for_cultivation(
        counts, verbose=(verbosity >= 2)
    )
    total_computational_volume = (
        clifford_computational_vol + non_clifford_lattice_surgery_vol + cultivation_volume
    )

    # Scale the raw measurement depth
    scaled_measurement_depth = measurement_depth.depth * logical_timesteps_per_measurement

    # Calculate limiting depth
    # Guard against division by zero when n_fluid_ancilla is zero (or negative),
    # which produces zoo and leads to NaN errors during substitution/simplification.
    if n_fluid_ancilla <= 0:
        volume_limited_depth = sympy.oo
    else:
        volume_limited_depth = total_computational_volume / n_fluid_ancilla  # type: ignore[operator]
    total_depth = sympy.Max(scaled_measurement_depth, volume_limited_depth)

    # Calculate total Clifford volume including idling
    idling_volume = n_algorithmic_qubits * total_depth
    total_clifford_volume = clifford_computational_vol + idling_volume

    # Calculate total spacetime volume
    total_spacetime_volume = total_computational_volume + idling_volume

    # Calculate total T count
    total_t_count = (
        counts.t
        + counts.toffoli * 4
        + counts.and_gate * 4
        + MIXED_FALLBACK_T_COUNT * (counts.z_rotation + counts.x_rotation)
    )

    # Calculate total rotation count
    total_rotation_count = counts.x_rotation + counts.z_rotation

    # Create and return the summary object
    summary = FLASQSummary(
        total_clifford_volume=total_clifford_volume,
        total_depth=total_depth,
        n_algorithmic_qubits=n_algorithmic_qubits,
        n_fluid_ancilla=n_fluid_ancilla,
        total_t_count=total_t_count,
        total_rotation_count=total_rotation_count,
        # Pass new fields
        measurement_depth_val=measurement_depth.depth,
        volume_limited_depth=volume_limited_depth,
        scaled_measurement_depth=scaled_measurement_depth,
        total_computational_volume=total_computational_volume,
        idling_volume=idling_volume,
        clifford_computational_volume=clifford_computational_vol,
        non_clifford_lattice_surgery_volume=non_clifford_lattice_surgery_vol,
        cultivation_volume=cultivation_volume,
        total_spacetime_volume=total_spacetime_volume,
    )

    if assumptions is not None:
        summary = summary.resolve_symbols(assumptions)

    if verbosity >= 1:
        print("\n" + "=" * 20 + " FLASQ Summary " + "=" * 20)
        print(f"Total Spacetime Volume: {summary.total_spacetime_volume}")
        print(f"Total T-gate Count: {summary.total_t_count}")
        print(f"Total Clifford Volume: {summary.total_clifford_volume}")
        print(f"Total Depth: {summary.total_depth}")
        print(f"Number of Algorithmic Qubits: {summary.n_algorithmic_qubits}")
        print(f"Number of Fluid Ancilla: {summary.n_fluid_ancilla}")
        print("=" * 55 + "\n")

    return summary


def get_rotation_depth(rotation_error: Optional[SymbolicFloat] = None) -> SymbolicFloat:
    """Returns the expected T-count via mixed fallback synthesis.

    T gates in rotation synthesis are sequential, so T-count = T-depth.

    Args:
        rotation_error: If provided, the ROTATION_ERROR symbol in
            MIXED_FALLBACK_T_COUNT will be substituted with this value.

    Returns:
        The calculated rotation depth, potentially symbolic.
    """
    depth_val = MIXED_FALLBACK_T_COUNT

    if rotation_error is not None:
        depth_val = substitute_until_fixed_point(
            depth_val, frozendict({ROTATION_ERROR: rotation_error}), try_make_number=True
        )

    return depth_val


# --- Pre-instantiated FLASQCostModel Instances ---

#: Conservative cost model using default parameters from the paper.
conservative_FLASQ_costs = FLASQCostModel()

#: Optimistic cost model with reduced overheads from the paper.
optimistic_FLASQ_costs = FLASQCostModel(
    # Base parameters for the optimistic model
    h_volume=1.5,
    s_volume=1.5,
    t_cultivation_volume=V_CULT_FACTOR,
    t_clifford_volume=T_REACT + 2.5,
    cnot_base_volume=0.0,
    cz_base_volume=0.0,
    connect_span_volume=1.0,
    compute_span_volume=1.0,
    toffoli_clifford_volume=5 * T_REACT + 39.0,
    and_clifford_volume=2 * T_REACT + 36.0,
    and_dagger_clifford_volume=0.0,
    # Optimistic Rotation Parameters (d=1, App A.1.8):
    extra_cost_per_t_gate_in_rotation=1.0,
    # Calculation: (7/6)*1.5 + (5/6)*1.5 + 2*(2*1) + 5 = 3 + 4 + 5 = 12.0
    extra_cost_per_rotation=12.0,
)
