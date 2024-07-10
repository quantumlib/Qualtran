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

import attrs
from typing import Sequence, Callable, Union
import math
import numpy as np
from qualtran.surface_code.algorithm_summary import AlgorithmSummary
from qualtran.surface_code.data_block import FastDataBlock, DataBlock
from qualtran.surface_code.quantum_error_correction_scheme_summary import (
    QuantumErrorCorrectionSchemeSummary,
)
from qualtran.surface_code.rotation_cost_model import RotationCostModel
from qualtran.surface_code.workflow import PhysicalResourceEstimationWorkflow, PhysicalEstimationParameters
from qualtran.surface_code.physical_cost import PhysicalCost
from qualtran.surface_code.magic_state_factory import MagicStateFactory
from qualtran.surface_code.multi_factory import MultiFactory
from qualtran.surface_code.ccz2t_cost_model import GidneyFowlerCCZ
from qualtran.surface_code.fifteen_to_one import FifteenToOne733, FifteenToOne933

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
    if alg.rotation_gates > 0:
        rotation_cost = rotation_model.rotation_cost(eps_syn / alg.rotation_gates)
        c_min += math.ceil(
            alg.rotation_circuit_depth * (rotation_cost.n_t + 4 * rotation_cost.n_ccz)
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
    q = FastDataBlock.grid_size(n_algo_qubits=int(alg.algorithm_qubits))
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
    total_magic = alg.to_magic_count(rotation_model=rotation_model, error_budget=eps_syn)
    return total_magic.n_t + 4 * total_magic.n_ccz


def _cost_at_num_timesteps(t: float, algorithm_summary: AlgorithmSummary, params: PhysicalEstimationParameters) -> PhysicalCost:
    cd = code_distance(
        error_budget=params.error_budget,
        time_steps=t,
        alg=algorithm_summary,
        qec=params.qec,
        physical_error_rate=params.physical_error_rate,
    )
    runtime_seconds = params.qec.error_detection_circuit_time_us(cd) * 1e-6 * t
    logical_qubits = FastDataBlock.grid_size(n_algo_qubits=algorithm_summary.algorithm_qubits)

    return PhysicalCost(
        data_block=params.data_block,
        code_distance=cd,
        duration_hr=runtime_seconds / 3600,
        footprint=logical_qubits*params.qec.physical_qubits(cd) + 
        params.magic_state_factory.footprint() * params.num_magic_factories,
        distillation_qubits_contrib=params.magic_state_factory.footprint()*params.num_magic_factories,
        logical_qubits_contib=logical_qubits*params.qec.physical_qubits(cd),
        magic_state_factory=MultiFactory(params.magic_state_factory, params.num_magic_factories),
        failure_prob=1e-6, # some value
    )


class AzureWorkflow(PhysicalResourceEstimationWorkflow):

    def minimum_runtime(
        self, algorithm_summary: AlgorithmSummary, params: PhysicalEstimationParameters
    ) -> PhysicalCost:
        return self.data_points(algorithm_summary, params)[-1]

    def minimum_qubits(
        self, algorithm_summary: AlgorithmSummary, params: PhysicalEstimationParameters
    ) -> PhysicalCost:
        return self.data_points(algorithm_summary, params)[0]

    def data_points(
        self, algorithm_summary: AlgorithmSummary, params: PhysicalEstimationParameters
    ) -> Sequence[PhysicalCost]:
        c_min = minimum_time_steps(params.error_budget, algorithm_summary, params.rotation_model)
        factory = MultiFactory(params.magic_state_factory, params.num_magic_factories)
        magic_count = algorithm_summary.to_magic_count(params.rotation_model, params.error_budget)
        factory_cycles = factory.n_cycles(magic_count, params.physical_error_rate)
        min_num_factories = int(np.ceil(factory_cycles / c_min))
        magic_counts = list(
            1 + np.random.choice(min_num_factories, replace=False, size=min(min_num_factories, 5))
        )
        magic_counts.sort(reverse=True)
        magic_counts = np.array(magic_counts)
        time_steps = np.ceil(factory_cycles / magic_counts)
        magic_counts[0] = min_num_factories
        time_steps[0] = c_min
        return [_cost_at_num_timesteps(t, algorithm_summary, attrs.evolve(params, magic_state_factory=factory, num_magic_factories=c)) for t, c in zip(time_steps, magic_counts)]
 
    def estimate(
        self,
        algorithm_summary: AlgorithmSummary,
        params: PhysicalEstimationParameters,
        objective: Callable[[PhysicalCost], float],
    ) -> Sequence[PhysicalCost]:
        return NotImplemented
    
    def supported_magic_state_factories(
        self,
    ) -> Sequence[Union[MagicStateFactory, type[MagicStateFactory]]]:
        """Returns a list of magic state factories that are supported by this workflow."""
        return [
            GidneyFowlerCCZ, FifteenToOne733, FifteenToOne933
        ]

    def supported_datablocks(self) -> Sequence[Union[DataBlock, type[DataBlock]]]:
        """Returns a list of data blocks that are supported by this workflow."""
        return [FastDataBlock]
