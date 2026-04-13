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

from qualtran.surface_code.flasq.optimization.analysis import (
    analyze_logical_circuit,
    calculate_single_flasq_summary,
    generate_circuit_specific_configs,
    generate_configs_for_constrained_qec,
)
from qualtran.surface_code.flasq.optimization.configs import (
    CoreParametersConfig,
    ErrorBudget,
    generate_configs_for_specific_cultivation_assumptions,
    generate_configs_from_cultivation_data,
)
from qualtran.surface_code.flasq.optimization.postprocessing import (
    post_process_for_failure_budget,
    post_process_for_logical_depth,
    post_process_for_pec_runtime,
)
from qualtran.surface_code.flasq.optimization.sweep import SweepResult, run_sweep

# Re-export these for backward compatibility and tests
from qualtran.surface_code.flasq.cirq_interop import convert_circuit_for_flasq_analysis
from qualtran.surface_code.flasq.error_mitigation import calculate_failure_probabilities
from qualtran.surface_code.flasq import cultivation_analysis

__all__ = [
    "CoreParametersConfig",
    "ErrorBudget",
    "SweepResult",
    "analyze_logical_circuit",
    "calculate_single_flasq_summary",
    "generate_circuit_specific_configs",
    "generate_configs_for_constrained_qec",
    "generate_configs_for_specific_cultivation_assumptions",
    "generate_configs_from_cultivation_data",
    "post_process_for_failure_budget",
    "post_process_for_logical_depth",
    "post_process_for_pec_runtime",
    "run_sweep",
    "convert_circuit_for_flasq_analysis",
    "calculate_failure_probabilities",
    "cultivation_analysis",
]
