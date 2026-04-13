"""qualtran_flasq: FLASQ cost model for fault-tolerant quantum resource estimation."""

from qualtran_flasq.cirq_interop import convert_circuit_for_flasq_analysis
from qualtran_flasq.error_mitigation import (
    calculate_error_mitigation_metrics,
    calculate_failure_probabilities,
)
from qualtran_flasq.flasq_model import (
    FLASQCostModel,
    FLASQSummary,
    apply_flasq_cost_model,
    conservative_FLASQ_costs,
    get_rotation_depth,
    optimistic_FLASQ_costs,
)
from qualtran_flasq.measurement_depth import TotalMeasurementDepth
from qualtran_flasq.naive_grid_qubit_manager import NaiveGridQubitManager
from qualtran_flasq.optimization import (
    ErrorBudget,
    generate_circuit_specific_configs,
    generate_configs_for_constrained_qec,
    generate_configs_from_cultivation_data,
    post_process_for_failure_budget,
    post_process_for_logical_depth,
    post_process_for_pec_runtime,
    run_sweep,
)
from qualtran_flasq.span_counting import TotalSpanCost
from qualtran_flasq.symbols import (
    MIXED_FALLBACK_T_COUNT,
    ROTATION_ERROR,
    T_REACT,
    V_CULT_FACTOR,
)
from qualtran_flasq.utils import substitute_until_fixed_point
from qualtran_flasq.volume_counting import FLASQGateTotals
from qualtran_flasq import cultivation_analysis  # noqa: F401 — imported as module

__all__ = [
    "FLASQCostModel",
    "FLASQSummary",
    "apply_flasq_cost_model",
    "conservative_FLASQ_costs",
    "optimistic_FLASQ_costs",
    "get_rotation_depth",
    "FLASQGateTotals",
    "TotalSpanCost",
    "TotalMeasurementDepth",
    "convert_circuit_for_flasq_analysis",
    "NaiveGridQubitManager",
    "ErrorBudget",
    "run_sweep",
    "generate_configs_for_constrained_qec",
    "generate_configs_from_cultivation_data",
    "generate_circuit_specific_configs",
    "post_process_for_failure_budget",
    "post_process_for_pec_runtime",
    "post_process_for_logical_depth",
    "calculate_error_mitigation_metrics",
    "calculate_failure_probabilities",
    "MIXED_FALLBACK_T_COUNT",
    "ROTATION_ERROR",
    "V_CULT_FACTOR",
    "T_REACT",
    "substitute_until_fixed_point",
    "cultivation_analysis",
]
