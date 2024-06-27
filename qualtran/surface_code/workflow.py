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

import abc
from typing import Callable, Sequence, Union

from attrs import frozen

from qualtran.surface_code.algorithm_summary import AlgorithmSummary
from qualtran.surface_code.data_block import DataBlock
from qualtran.surface_code.magic_state_factory import MagicStateFactory
from qualtran.surface_code.physical_cost import PhysicalCost
from qualtran.surface_code.quantum_error_correction_scheme_summary import (
    QuantumErrorCorrectionSchemeSummary,
)
from qualtran.surface_code.rotation_cost_model import RotationCostModel


@frozen
class PhysicalEstimationParameters:
    """Parameters for the physical estimation workflow.

    Physical resource estimation workflows need to know information such as the
    physical error rate, the error budget, magic state factory, and other choices
    to calculate the physical cost.

    Attributes:
        physical_error_rate: The physical error rate.
        error_budget: The error budget.
        qec: The quantum error correction scheme.
        magic_state_factory: A magic state factory instance or class.
        num_magic_factories: The number of magic state factories.
        data_block: A data block instance or class.
        rotation_model: The rotation cost model.
    """

    physical_error_rate: float

    error_budget: float

    qec: QuantumErrorCorrectionSchemeSummary

    magic_state_factory: Union[MagicStateFactory, type[MagicStateFactory]]
    num_magic_factories: int

    data_block: Union[DataBlock, type[DataBlock]]

    rotation_model: RotationCostModel


class PhysicalResourceEstimationWorkflow(abc.ABC):
    """Abstract base class for physical resource estimation workflows."""

    @abc.abstractmethod
    def minimum_runtime(
        self, algorithm_summary: AlgorithmSummary, params: PhysicalEstimationParameters
    ) -> PhysicalCost:
        """Performs estimation with an objective to minimize the runtime."""

    @abc.abstractmethod
    def minimum_qubits(
        self, algorithm_summary: AlgorithmSummary, params: PhysicalEstimationParameters
    ) -> PhysicalCost:
        """Performs estimation with an objective to minimize the number of qubits."""

    @abc.abstractmethod
    def data_points(
        self, algorithm_summary: AlgorithmSummary, params: PhysicalEstimationParameters
    ) -> Sequence[PhysicalCost]:
        """Returns an ordered sequence of physical costs that go from slowest (minimum qubits) to fastest (maximum qubits)."""

    @abc.abstractmethod
    def estimate(
        self,
        algorithm_summary: AlgorithmSummary,
        params: PhysicalEstimationParameters,
        objective: Callable[[PhysicalCost], float],
    ) -> Sequence[PhysicalCost]:
        """Performs estimation with a custom objective function."""

    @abc.abstractmethod
    def supported_magic_state_factories(
        self,
    ) -> Sequence[Union[MagicStateFactory, type[MagicStateFactory]]]:
        """Returns a list of magic state factories that are supported by this workflow."""

    @abc.abstractmethod
    def supported_datablocks(self) -> Sequence[Union[DataBlock, type[DataBlock]]]:
        """Returns a list of data blocks that are supported by this workflow."""
