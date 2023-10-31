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

import abc
import math

from attrs import frozen

from qualtran.surface_code.algorithm_summary import AlgorithmSummary


class RotationCostModel(abc.ABC):
    """Analytical estimate of the complexity of approximating a rotation given an error budget."""

    @abc.abstractmethod
    def rotation_cost(self, error_budget: float) -> AlgorithmSummary:
        """Cost of a single rotation."""

    @abc.abstractmethod
    def prepartion_overhead(self, error_budget) -> AlgorithmSummary:
        """Cost of preparation circuit."""


@frozen
class RotationLogarithmicModel(RotationCostModel):
    r"""A linear model in the log of the error budget with no preparation cost.

    $$
    \#T = -\textrm{slope} \log_2{\textrm{budget}} + \textrm{overhead}
    $$

    Attributes:
        slope: The coefficient of $log_2{budget}$.
        overhead: The overhead.
        gateset: A human-readable description of the gate set (e.g. 'Clifford+T').
        approximation_protocol: A description or reference to the approximation protocol
            (e.g. `Diagonal: https://arxiv.org/abs/1403.2975`)
        reference: A human-readable description of the source of the model
            (e.g. 'https://arxiv.org/abs/1404.5320').
    """
    slope: float
    overhead: float
    gateset: str | None = None
    approximation_protocol: str | None = None
    reference: str | None = None

    def rotation_cost(self, error_budget: float) -> AlgorithmSummary:
        return AlgorithmSummary(
            t_gates=math.ceil(-self.slope * math.log2(error_budget) + self.overhead)
        )

    def prepartion_overhead(self, error_budget) -> AlgorithmSummary:
        return AlgorithmSummary()


@frozen
class ConstantWithOverheadRotationCost(RotationCostModel):
    r"""A rotation cost of bitsize - 2 toffoli per rotation independent of the error budget.

    This model assumes a state $\ket{\phi}$ has been prepared using a standard technique, then
    each rotation is applied with bitsize digits of accuracy using bitsize - 2 Toffoli gates.
    $$
    \ket{\phi} = \frac{1}{\sqrt{2^{b}}} \sum_{k=0}^{2^b-1} e^{-2\pi i k/2^b} \ket{k}
    $$
    Where $b$ is the bitsize/number of digits of accuracy.

    reference: https://doi.org/10.1103/PRXQuantum.1.020312

    Attributes:
        bitsize: Number of digits of accuracy for approximating a rotation.
        overhead_rotation_cost: The cost model of preparing the initial rotation.
        reference: A human-readable description of the source of the model
            (e.g. 'https://arxiv.org/abs/1404.5320').
    """

    bitsize: int
    overhead_rotation_cost: RotationCostModel
    reference: str | None = None

    def rotation_cost(self, error_budget: float) -> AlgorithmSummary:
        return AlgorithmSummary(toffoli_gates=max(self.bitsize - 2, 0))

    def prepartion_overhead(self, error_budget) -> AlgorithmSummary:
        return self.bitsize * self.overhead_rotation_cost.rotation_cost(error_budget / self.bitsize)


BeverlandEtAlRotationCost = RotationLogarithmicModel(
    slope=0.53,
    overhead=5.3,
    gateset='Clifford+T',
    approximation_protocol='Mixed fallback:https://arxiv.org/abs/2203.10064',
    reference='https://arxiv.org/abs/2211.07629:D2',
)
