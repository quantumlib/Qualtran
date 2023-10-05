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
import enum

import numpy as np
from attrs import field, frozen

from qualtran.surface_code.physical_parameters import PhysicalParameters


@frozen
class QuantumErrorCorrectionSchemeSummary(abc.ABC):
    """QuantumErrorCorrectionSchemeSummary represents a high level view of a QEC scheme.

        QuantumErrorCorrectionSchemeSummary provides estimates for the logical error rate,
        number of physical qubits and the logical time step given a code distance and
        physical assumptions.

        The logical error rate as a function of code distance $d$ and physical error rate $p$ 
        is given by
        $$
        a \left ( \frac{p}{p^*}  \right )^\frac{d + 1}{2}
        $$
        Where $a$ is the error_rate_scaler and $p^*$ is the error_rate_threshold.

    Attributes:
        error_rate_scaler: Logical error rate coefficient.
        error_rate_threshold: Logical error rate threshold.
        reference: source of the estimates.
    """

    error_rate_scaler: float = field(default=0.1, repr=lambda x: f'{x:g}')
    error_rate_threshold: float = field(default=0.01, repr=lambda x: f'{x:g}')
    reference: str | None = None

    def logical_error_rate(
        self, code_distance: int | np.ndarray, physical_error_rate: float | np.ndarray
    ) -> float | np.ndarray:
        """The logical error rate for the given code distance using this scheme."""
        return self.error_rate_scaler * np.power(
            physical_error_rate / self.error_rate_threshold, (code_distance + 1) / 2
        )

    @abc.abstractmethod
    def physical_qubits(self, code_distance: int | np.ndarray) -> int | np.ndarray:
        """The number of physical qubits for the given code distance using this scheme."""

    @abc.abstractmethod
    def logical_time_step(
        self, code_distance: int | np.ndarray, physical_parameters: PhysicalParameters
    ) -> float | np.ndarray:
        """The duration of a logical time step for the given code distance using this scheme."""


class GateBasedSurfaceCodeSource(enum.Enum):
    arXiv12080928 = 0  # Folwer Model.
    arXiv221107629 = 1  # Azure Model.


@frozen
class GateBasedSurfaceCode(QuantumErrorCorrectionSchemeSummary):
    """Gate Based Surface Code."""

    source: GateBasedSurfaceCodeSource = GateBasedSurfaceCodeSource.arXiv12080928

    def physical_qubits(self, code_distance: int | np.ndarray) -> int | np.ndarray:
        return 2 * code_distance**2

    def logical_time_step(
        self, code_distance: int | np.ndarray, physical_parameters: PhysicalParameters
    ) -> float | np.ndarray:
        measurement_coef = 1 if self.source is GateBasedSurfaceCodeSource.arXiv12080928 else 2
        return (
            4 * physical_parameters.t_gate_ns + measurement_coef * physical_parameters.t_meas_ns
        ) * code_distance


class MeasurementBasedSurfaceCode(QuantumErrorCorrectionSchemeSummary):
    """Measurement Based Surface Code."""

    def physical_qubits(self, code_distance: int | np.ndarray) -> int | np.ndarray:
        return 2 * code_distance**2

    def logical_time_step(
        self, code_distance: int | np.ndarray, physical_parameters: PhysicalParameters
    ) -> float | np.ndarray:
        return 20 * physical_parameters.t_meas_ns * code_distance


class MeasurementBasedHastingsHaahCode(QuantumErrorCorrectionSchemeSummary):
    """Measurement Based Hastings&Haah Code."""

    def physical_qubits(self, code_distance: int | np.ndarray) -> int | np.ndarray:
        return 4 * code_distance**2 + 8 * (code_distance - 1)

    def logical_time_step(
        self, code_distance: int | np.ndarray, physical_parameters: PhysicalParameters
    ) -> float | np.ndarray:
        return 3 * physical_parameters.t_meas_ns * code_distance
