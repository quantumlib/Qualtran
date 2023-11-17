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

import numpy as np
from attrs import field, frozen


@frozen
class QuantumErrorCorrectionSchemeSummary(abc.ABC):
    r"""QuantumErrorCorrectionSchemeSummary represents a high level view of a QEC scheme.

    QuantumErrorCorrectionSchemeSummary provides estimates for the logical error rate,
    number of physical qubits and the time of an error detection cycle.

    The logical error rate as a function of code distance $d$ and physical error rate $p$
    is given by
    $$
    a \left ( \frac{p}{p^*}  \right )^\frac{d + 1}{2}
    $$
    Where $a$ is the error_rate_scaler and $p^*$ is the error_rate_threshold.

    Note: The logical error-suppression factor $\Lambda = \frac{p^*}{p}$

    Attributes:
        error_rate_scaler: Logical error rate coefficient.
        error_rate_threshold: Logical error rate threshold.
        reference: source of the estimates in human readable format.
    """

    error_rate_scaler: float = field(repr=lambda x: f'{x:g}')
    error_rate_threshold: float = field(repr=lambda x: f'{x:g}')
    reference: str | None

    def logical_error_rate(
        self, code_distance: int, physical_error_rate: float | np.ndarray
    ) -> float | np.ndarray:
        """The logical error rate given the physical error rate."""
        return self.error_rate_scaler * np.power(
            physical_error_rate / self.error_rate_threshold, (code_distance + 1) / 2
        )

    @abc.abstractmethod
    def physical_qubits(self, code_distance: int) -> int:
        """The number of physical qubits used by the error correction circuit."""

    @abc.abstractmethod
    def error_detection_circuit_time_us(self, code_distance: int) -> float:
        """The time of a quantum error correction cycle in seconds."""


@frozen
class SimpliedSurfaceCode(QuantumErrorCorrectionSchemeSummary):
    """A Surface Code Quantum Error Correction Scheme.

    Attributes:
        single_stabilizer_time_us: Max time of a single X or Z stabilizer measurement.
    """

    single_stabilizer_time_us: float

    def physical_qubits(self, code_distance: int) -> int:
        return 2 * code_distance**2

    def error_detection_circuit_time_us(self, code_distance: int) -> float:
        """Equals the time to measure a stabilizer times the depth of the circuit."""
        return self.single_stabilizer_time_us * code_distance


BeverlandSuperConductingQubits = SimpliedSurfaceCode(
    error_rate_scaler=0.03,
    error_rate_threshold=0.01,
    single_stabilizer_time_us=0.4,  # Equals 4*t_gate+2*t_meas where t_gate=50ns and t_meas=100ns.
    reference='https://arxiv.org/abs/2211.07629',
)
