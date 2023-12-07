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

from qualtran.surface_code.algorithm_summary import AlgorithmSummary
from qualtran.surface_code.quantum_error_correction_scheme_summary import (
    QuantumErrorCorrectionSchemeSummary,
)
from qualtran.surface_code.rotation_cost_model import RotationCostModel


def logical_qubits(algorithm_specs: AlgorithmSummary) -> int:
    r"""Number of logical qubits needed for the algorithm.

    Equals:
    $$
        2 Q_\mathrm{alg} + \lceil \sqrt{8 Q_\mathrm{alg}} \rceil + 1
    $$

    Source: Equation D1 in https://arxiv.org/abs/2211.07629.

    Args:
        algorithm_specs: A summary of an algorithm/circuit.
    """
    q_alg = algorithm_specs.algorithm_qubits
    return math.ceil(2 * q_alg + math.sqrt(8 * q_alg) + 1)


def minimum_time_steps(
    error_budget: float, alg: AlgorithmSummary, rotation_model: RotationCostModel
) -> int:
    r"""Minimum number of time steps needed for the algorithm.

    Equals
    $$
        M_\mathrm{meas} + M_R + M_T + 3 M_\mathrm{Tof} + D_R \textrm{rotation cost}
    $$
    Where:
        $M_\mathrm{meas}$ is the number of measurements.
        $M_R$ is the number of rotations.
        $M_T$ is the number of T operations.
        $M_mathrm{Tof}$ is the number of toffoli operations.
        $D_R$ is the depth of the rotation circuit.
        $\textrm{rotation cost}$ is the number of T operations needed to approximate a rotation to $\epsilon/(3*M_R)$.
    Source: Equation D3 in https://arxiv.org/abs/2211.07629.

    Args:
        error_budget: Error Budget.
        alg: A summary of an algorithm/circuit.
        rotation_model: Cost model used to compute the number of T gates
            needed to approximate rotations.
    """
    c_min = math.ceil(alg.measurements + alg.rotation_gates + alg.t_gates + 3 * alg.toffoli_gates)
    eps_syn = error_budget / 3
    c_min += math.ceil(
        alg.rotation_circuit_depth
        * rotation_model.rotation_cost(eps_syn / alg.rotation_gates).t_gates
    )
    return c_min


def code_distance(
    error_budget: float,
    time_steps: float,
    alg: AlgorithmSummary,
    qec: QuantumErrorCorrectionSchemeSummary,
    physical_error_rate: float,
) -> int:
    r"""Minimum code distance needed to run the algorithm within the error budget.

    This is the code distance $d$ that satisfies $QCP = \epsilon/3$. Where:
        $\epsilon$ is the error budget.
        Q is the number of logical qubits.
        C is the number of time steps.
        P(d) is the logical error rate.

    Args:
        error_budget: Error Budget.
        time_steps: Number of time steps used to run the algorithm.
        alg: A summary of an algorithm/circuit.
        qec: Quantum Error Correction Scheme.
        physical_error_rate: The physical error rate of the device.
    """
    q = logical_qubits(alg)
    return qec.code_distance_from_budget(physical_error_rate, error_budget / (3 * q * time_steps))


def t_states(
    error_budget: float, alg: AlgorithmSummary, rotation_model: RotationCostModel
) -> float:
    r"""Total number of T states consumed by the algorithm.

    Equals
    $$
       M_T + 4 M_\mathrm{Tof} + M_R \textrm{rotation cost}
    $$
    Where:
        $M_R$ is the number of rotations.
        $M_T$ is the number of T operations.
        $M_mathrm{Tof}$ is the number of toffoli operations.
        $\textrm{rotation cost}$ is the number of T operations needed to approximate a rotation to $\epsilon/(3*M_R)$.
    Source: D4 in https://arxiv.org/abs/2211.07629.

    Args:
        error_budget: Error Budget.
        alg: A summary of an algorithm/circuit.
        rotation_model: Cost model used to compute the number of T gates
            needed to approximate rotations.
    """
    eps_syn = error_budget / 3
    return (
        alg.t_gates
        + 4 * alg.toffoli_gates
        + alg.rotation_gates * rotation_model.rotation_cost(eps_syn / alg.rotation_gates).t_gates
    )
