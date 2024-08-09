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
from typing import TYPE_CHECKING

import attrs

from qualtran.resource_counting import GateCounts
from qualtran.symbolics import SymbolicInt

if TYPE_CHECKING:
    from qualtran.surface_code import AlgorithmSummary, QECScheme, RotationCostModel


def minimum_time_steps(
    *, error_budget: float, alg: 'AlgorithmSummary', rotation_model: 'RotationCostModel'
) -> int:
    r"""Minimum number of time steps needed for the algorithm.

    Equals
    $$
        M_\mathrm{meas} + M_R + M_T + 3 M_\mathrm{Tof} + D_R \textrm{rotation cost}
    $$

    Where:
     - $M_\mathrm{meas}$ is the number of measurements.
     - $M_R$ is the number of rotations.
     - $M_T$ is the number of T operations.
     - $M_mathrm{Tof}$ is the number of toffoli operations.
     - $D_R$ is the number of layers containing at least one rotation. This can be smaller than
       the total number of non-Clifford layers since it excludes layers consisting only of T or
       Toffoli gates.
     - $\textrm{rotation cost}$ is the number of T operations needed to approximate a rotation to $\epsilon/(3*M_R)$.

    Reference:
        https://arxiv.org/abs/2211.07629.
        Equation D3.

    Args:
        error_budget: The total error budget. One third is prescribed to be used for rotation
            synthesis.
        alg: The logical algorithm costs, from which we extract the gate count.
        rotation_model: Cost model used to compute the number of T gates
            needed to approximate rotations.
    """
    M = alg.n_logical_gates.total_beverland_count()
    c_min = M['meas'] + M['R'] + M['T'] + 3 * M['Tof']
    eps_syn = error_budget / 3
    if M['R'] > 0:
        # Note: The argument to the rotation_cost method is inverted relative to the notation in
        # eq. D3. The log rotation model (corresponding to eq. D3) has a negative sign outside the
        # log.
        rot_err_budget = eps_syn / M['R']
        rotation_cost = rotation_model.rotation_cost(
            rot_err_budget
        ) + rotation_model.preparation_overhead(rot_err_budget)

        if alg.n_rotation_layers is not None:
            # We don't actually push all the cliffords out and count the number of
            # rotation layers, so this is just the number of rotations $M_R$ by default.
            # If you are trying to reproduce numbers exactly, you can provide an explicit
            # number of rotation layers.
            M['D_R'] = alg.n_rotation_layers
        c_min += math.ceil(M['D_R'] * (rotation_cost.t + 4 * rotation_cost.toffoli))
    return c_min


def code_distance(
    *,
    error_budget: float,
    time_steps: float,
    alg: 'AlgorithmSummary',
    qec_scheme: 'QECScheme',
    physical_error: float,
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
        alg: The logical algorithm costs, from which we extract the gate count.
        qec_scheme: Quantum Error Correction Scheme.
        physical_error: The physical error rate of the device.
    """
    from qualtran.surface_code import FastDataBlock

    q = FastDataBlock.get_n_tiles(n_algo_qubits=alg.n_algo_qubits)
    return qec_scheme.code_distance_from_budget(physical_error, error_budget / (3 * q * time_steps))


def n_discrete_logical_gates(
    *, eps_syn: float, alg: 'AlgorithmSummary', rotation_model: 'RotationCostModel'
) -> GateCounts:
    r"""Total number of T and CCZ states after synthesizing rotations.

    Args:
        eps_syn: The error budget for synthesizing rotations.
        alg: The logical algorithm costs, from which we extract the gate count.
        rotation_model: Cost model used to compute the number of T gates
            needed to approximate rotations.
    """
    n_rotations: SymbolicInt = alg.n_logical_gates.rotation
    ret = attrs.evolve(alg.n_logical_gates, rotation=0)
    if n_rotations > 0:
        ret = (
            ret
            + rotation_model.preparation_overhead(eps_syn)
            + n_rotations * rotation_model.rotation_cost(eps_syn / n_rotations)
        )
    return ret


def t_states(
    *, error_budget: float, alg: 'AlgorithmSummary', rotation_model: 'RotationCostModel'
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
    return n_discrete_logical_gates(
        eps_syn=eps_syn, alg=alg, rotation_model=rotation_model
    ).total_t_count()
