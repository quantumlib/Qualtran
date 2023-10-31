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
class TFactory(MagicStateFactory):
    """A magic state factory for T states.

    Attributes:
        num_qubits: Number of qubits used by the factory.
        duration: Time taken by the factory to produce T states.
        t_states_rate: Number of T states per production cycle.
        reference: Source of these estimates.
    """

    num_qubits: int
    generation_cycle_duration_ns: float = field(repr=lambda x: f'{x:g}')
    num_t_per_cycle: float = field(repr=lambda x: f'{x:g}')
    error_rate: float = field(repr=lambda x: f'{x:g}')
    reference: str | None = None

    def footprint(self) -> int:
        return self.num_qubits

    def n_cycles(self, n_magic: AlgorithmSummary) -> int:
        t_states = n_magic.t_gates + 4 * n_magic.toffoli_gates
        return math.ceil(t_states / self.num_t_per_cycle)

    def distillation_error(self, n_magic: AlgorithmSummary, phys_err: float) -> float:
        return NotImplemented

    def spacetime_footprint(self) -> float:
        time_per_t_state = self.generation_cycle_duration_ns / self.num_t_per_cycle
        return time_per_t_state * self.num_qubits
