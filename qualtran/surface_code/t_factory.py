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
from typing import Optional

from attrs import field, frozen

from qualtran.surface_code.algorithm_summary import AlgorithmSummary
from qualtran.surface_code.magic_state_factory import MagicStateFactory


@frozen
class Simple15to1TFactory(MagicStateFactory):
    """A summary of properties of a T-states factory.

    Represents 15-to-1 T-state factories as described in Appendix C of https://arxiv.org/abs/2211.07629
    The number of rounds and their construction (code distance, number of units, ...etc) is ignored
    and only the overall specification is reported as per table VII.

    Attributes:
        num_qubits: Number of physical qubits used by the factory.
        error_rate: Probability of not accepting a magic state
        reference: Source of these estimates.
    """

    num_qubits: int
    error_rate: float = field(repr=lambda x: f'{x:g}')
    reference: Optional[str] = None

    def footprint(self) -> int:
        return self.num_qubits

    def n_cycles(self, n_magic: AlgorithmSummary) -> int:
        t_states = n_magic.t_gates + 4 * n_magic.toffoli_gates
        # Number of cycles equals to number of T states since the factory creates 1 state per cycle
        return t_states

    def distillation_error(self, n_magic: AlgorithmSummary, phys_err: float) -> float:
        t_states = n_magic.t_gates + 4 * n_magic.toffoli_gates
        return t_states * self.error_rate
