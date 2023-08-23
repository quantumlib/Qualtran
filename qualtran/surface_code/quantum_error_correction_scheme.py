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

import attr
import numpy as np
from attrs import frozen

from qualtran.surface_code.physical_parameters import PhysicalParameters


@frozen
class QuantumErrorCorrectionScheme:
    """QuantumErrorCorrectionScheme represents a quantum error correction scheme.

        QuantumErrorCorrectionScheme provides estimates for the logical error rate,
        number of physical qubits and the logical time step given a code distance and
        physical assumptions.

    Attributes:
        error_rate_scaler: Logical error rate coefficient.
        error_rate_threshold: Logical error rate threshold.
        reference: source of the estimates.
    """

    error_rate_scaler = attr.ib(type=float, default=0.1, repr=lambda x: f'{x:g}')
    error_rate_threshold = attr.ib(type=float, default=0.01, repr=lambda x: f'{x:g}')
    reference = attr.ib(type=str, default='')

    def logical_error_rate(
        self, code_distance: int | np.ndarray, physical_error_rate: float | np.ndarray
    ) -> float | np.ndarray:
        """Computes the logical error rate."""
        return self.error_rate_scaler * np.power(
            physical_error_rate / self.error_rate_threshold, (code_distance + 1) / 2
        )

    def physical_qubits(self, code_distance: int | np.ndarray) -> int | np.ndarray:
        """Computes number of physical qubits"""

    def logical_time_step(
        self, code_distance: int | np.ndarray, physical_parameters: PhysicalParameters
    ) -> float:
        """Computes the logical time step."""


class GateBasedSurfaceCode(QuantumErrorCorrectionScheme):
    """Gate Based Surface Code."""

    def physical_qubits(self, code_distance: int | np.ndarray) -> int | np.ndarray:
        return 2 * code_distance**2

    def logical_time_step(
        self, code_distance: int | np.ndarray, physical_parameters: PhysicalParameters
    ) -> float:
        return (4 * physical_parameters.t_gate + 2 * physical_parameters.t_meas) * code_distance


class MeasurementBasedSurfaceCode(QuantumErrorCorrectionScheme):
    """Measurement Based Surface Code."""

    def physical_qubits(self, code_distance: int | np.ndarray) -> int | np.ndarray:
        return 2 * code_distance**2

    def logical_time_step(
        self, code_distance: int | np.ndarray, physical_parameters: PhysicalParameters
    ) -> float:
        return 20 * physical_parameters.t_meas * code_distance


class MeasurementBasedHastingsHaahCode(QuantumErrorCorrectionScheme):
    """Measurement Based Hastings&Haah Code."""

    def physical_qubits(self, code_distance: int | np.ndarray) -> int | np.ndarray:
        return 4 * code_distance**2 + 8 * (code_distance - 1)

    def logical_time_step(
        self, code_distance: int | np.ndarray, physical_parameters: PhysicalParameters
    ) -> float:
        return 3 * physical_parameters.t_meas * code_distance
