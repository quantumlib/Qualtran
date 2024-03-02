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
from typing import Optional

from attrs import field, frozen


@frozen
class QuantumErrorCorrectionSchemeSummary(abc.ABC):
    r"""QuantumErrorCorrectionSchemeSummary represents a high-level view of a QEC scheme.

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
        reference: source of the estimates in a human-readable format.
    """

    error_rate_scaler: float = field(repr=lambda x: f'{x:g}')
    error_rate_threshold: float = field(repr=lambda x: f'{x:g}')
    reference: Optional[str] = None

    def logical_error_rate(self, code_distance: int, physical_error_rate: float) -> float:
        """Logical error suppressed with code distance for this physical error rate.

        This is an estimate, see the references section.

        The formula was originally expressed as $p_l = a (b * p_p)^((d+1)/2)$ physical
        error rate $p_p$ and parameters $a$ and $b$. This can alternatively be expressed with
        $p_th = (1/b)$ roughly corresponding to the code threshold. This is sometimes also
        expressed with $lambda = p_th / p_p$. A lambda of 10, for example, would be p_p = 1e-3
        and p_th = 0.01. The pre-factor $a$ has no clear provenance.

        References:
            Low overhead quantum computation using lattice surgery. Fowler and Gidney (2018).
            https://arxiv.org/abs/1808.06709.
            See section XV for introduction of this formula, with citation to below.

            Surface code quantum error correction incorporating accurate error propagation.
            Fowler et. al. (2010). https://arxiv.org/abs/1004.0255.
            Note: this doesn't actually contain the formula from the above reference.
        """
        return self.error_rate_scaler * math.pow(
            physical_error_rate / self.error_rate_threshold, (code_distance + 1) / 2
        )

    def code_distance_from_budget(self, physical_error_rate: float, budget: float) -> int:
        """Get the code distance that keeps one below the logical error `budget`."""

        # See: `logical_error_rate()`. p_l = a Λ^(-r) where r = (d+1)/2
        # Which we invert: r = ln(p_l/a) / ln(1/Λ)
        r = math.log(budget / self.error_rate_scaler) / math.log(
            physical_error_rate / self.error_rate_threshold
        )
        d = 2 * math.ceil(r) - 1
        if d < 3:
            return 3
        return d

    @abc.abstractmethod
    def physical_qubits(self, code_distance: int) -> int:
        """The number of physical qubits per logical qubit used by the error detection circuit."""

    @abc.abstractmethod
    def error_detection_circuit_time_us(self, code_distance: int) -> float:
        """The time of a quantum error detection cycle in microseconds."""


@frozen
class SimpliedSurfaceCode(QuantumErrorCorrectionSchemeSummary):
    """A Surface Code Quantum Error Correction Scheme.

    Attributes:
        single_stabilizer_time_us: Max time of a single X or Z stabilizer measurement.
    """

    single_stabilizer_time_us: float = 1

    def physical_qubits(self, code_distance: int) -> int:
        return 2 * code_distance**2

    def error_detection_circuit_time_us(self, code_distance: int) -> float:
        """Equals the time to measure a stabilizer times the depth of the circuit."""
        return self.single_stabilizer_time_us * code_distance


BeverlandSuperconductingQubits = SimpliedSurfaceCode(
    error_rate_scaler=0.03,
    error_rate_threshold=0.01,
    single_stabilizer_time_us=0.4,  # Equals 4*t_gate+2*t_meas where t_gate=50ns and t_meas=100ns.
    reference='https://arxiv.org/abs/2211.07629',
)

FowlerSuperconductingQubits = SimpliedSurfaceCode(
    error_rate_scaler=0.1,
    error_rate_threshold=0.01,
    single_stabilizer_time_us=1,
    reference='https://arxiv.org/abs/1808.06709',
)

BeverlandMajoranaQubits = SimpliedSurfaceCode(
    error_rate_scaler=0.03,
    error_rate_threshold=0.01,
    single_stabilizer_time_us=0.6,  # Equals 4*t_gate+2*t_meas where t_gate=100ns and t_meas=100ns.
    reference='https://arxiv.org/abs/2211.07629',
)

BeverlandTrappedIonQubits = SimpliedSurfaceCode(
    error_rate_scaler=0.03,
    error_rate_threshold=0.01,
    single_stabilizer_time_us=600,  # Equals 4*t_gate+2*t_meas where t_gate=100us and t_meas=100us.
    reference='https://arxiv.org/abs/2211.07629',
)
