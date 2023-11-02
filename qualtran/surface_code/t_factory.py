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

from attrs import field, frozen

from qualtran.surface_code.algorithm_summary import AlgorithmSummary
from qualtran.surface_code.magic_state_factory import MagicStateFactory


@frozen
class SimpleTFactory(MagicStateFactory):
    """A summary of properties of a T-states factory.

    A summary of T-state factories as described in Appendix E of https://arxiv.org/abs/2211.07629

    Attributes:
        num_qubits: Number of physical qubits used by the factory.
        generation_time_ns: Time to generate a single T-state.
        distillation_error_: Probability of not accepting a magic state
        reference: Source of these estimates.
    """

    num_qubits: int
    generation_time_us: float = field(repr=lambda x: f'{x:g}')
    distillation_error_: float = field(repr=lambda x: f'{x:g}')
    reference: str | None = None

    def footprint(self) -> int:
        return self.num_qubits

    def n_cycles(self, n_magic: AlgorithmSummary) -> int:
        t_states = n_magic.t_gates + 4 * n_magic.toffoli_gates
        expected_cycles_per_t_state = 1 / (1 - self.distillation_error_)
        return math.ceil(t_states * expected_cycles_per_t_state)

    def distillation_error(self, n_magic: AlgorithmSummary, phys_err: float) -> float:
        t_states = n_magic.t_gates + 4 * n_magic.toffoli_gates
        return t_states * self.distillation_error_

    def spacetime_footprint(self) -> float:
        return self.generation_time_us * self.num_qubits
