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

import math

import numpy as np

from qualtran.surface_code.algorithm_specs import AlgorithmSpecs
from qualtran.surface_code.magic_state_factory import MagicStateCount
from qualtran.surface_code.physical_parameters import PhysicalParameters
from qualtran.surface_code.quantum_error_correction_scheme import QuantumErrorCorrectionScheme
from qualtran.surface_code.rotation_cost_model import RotationCostModel
from qualtran.surface_code.t_factory import TFactory


def logical_qubits(algorithm_specs: AlgorithmSpecs) -> int:
    """Number of logical qubits needed for the algorithm.

        Source: Equation D1 in https://arxiv.org/abs/2211.07629.

    Args:
        algorithm_specs: A summary of an algorithm/circuit.
    """
    q_alg = algorithm_specs.algorithm_qubits
    return 2 * q_alg + math.ceil(math.sqrt(8 * q_alg)) + 1


def minimum_time_steps(
    error_budget: float | np.ndarray, alg: AlgorithmSpecs, rotation_cost: RotationCostModel
) -> int | np.ndarray:
    """Minimum number of time steps needed for the algorithm.

        Source: Equation D3 in https://arxiv.org/abs/2211.07629.

    Args:
        error_budget: Error Budget.
        alg: A summary of an algorithm/circuit.
        rotation_cost: Cost model used to compute the number of T gates
            needed to approximate rotations.
    """
    c_min = alg.measurements + alg.rotation_gates + alg.t_gates + 3 * alg.toffoli_gates
    eps_syn = error_budget / 3
    c_min += alg.rotation_circuit_depth * np.ceil(
        rotation_cost.mean_cost(eps_syn / alg.rotation_gates)
    )
    return c_min


def code_distance(
    error_budget: float | np.ndarray,
    time_steps: float | np.ndarray,
    alg: AlgorithmSpecs,
    qec_scheme: QuantumErrorCorrectionScheme,
    physical_parameters: PhysicalParameters,
) -> int:
    """Minimum code distance needed to run the algorithm within the error budget.

        Source: -Corrected- E2 from https://arxiv.org/abs/2211.07629.

    Args:
        error_budget: Error Budget.
        time_steps: Number of time steps used to run the algorithm.
        alg: A summary of an algorithm/circuit.
        qec_scheme: Quantum Error Correction Scheme.
        physical_parameters: Physical Assumptions.
    """
    q = logical_qubits(alg)
    d = 2 * np.log((3 * qec_scheme.error_rate_scaler * q * time_steps) / error_budget)
    d /= np.log(qec_scheme.error_rate_threshold / physical_parameters.physical_error_rate)
    d -= 1
    d = np.clip(np.ceil(d), a_min=3, a_max=None)
    if isinstance(d, float):
        d = int(d)
    else:
        d = np.array(d, dtype=np.uint64)
    d += 1 - (d & 1)
    return d


def t_states(
    error_budget: float | np.ndarray, alg: AlgorithmSpecs, rotation_cost: RotationCostModel
) -> float | np.ndarray:
    """Total number of T states consumed by the algorithm.

        Source: D4 in https://arxiv.org/abs/2211.07629.

    Args:
        error_budget: Error Budget.
        alg: A summary of an algorithm/circuit.
        rotation_cost: Cost model used to compute the number of T gates
            needed to approximate rotations.
    """
    eps_syn = error_budget / 3
    return (
        alg.t_gates
        + 4 * alg.toffoli_gates
        + alg.rotation_gates * np.ceil(rotation_cost.mean_cost(eps_syn / alg.rotation_gates))
    )


def number_t_factories(
    num_t_states: float | np.ndarray, algorithm_runtime: float | np.ndarray, t_factory: TFactory
) -> float | np.ndarray:
    """Minimum number of T factories needed to run an algorithm.

        Source: E5 in https://arxiv.org/abs/2211.07629.

    Args:
        num_t_states: Total number of T states consumed by the algorithm.
        algorithm_runtime: Duration of the circuit runtime.
        t_factory: The T factory being used.
    """
    return np.ceil(t_factory.n_cycles(MagicStateCount(num_t_states, 0)) / algorithm_runtime)
