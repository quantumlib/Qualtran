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
from attrs import frozen

from qualtran.surface_code.magic_state_factory import MagicStateCount, MagicStateFactory

_PRETTY_FLOAT = attr.ib(type=float, default=0.0, repr=lambda x: f'{x:g}')


@frozen
class TFactory(MagicStateFactory):
    """TFactory represents a magic state factory for T states.

    Attributes:
        num_qubits: Number of qubits used by the factory.
        duration: Time taken by the factory to produce T states.
        t_states_rate: Number of T states per production cycle.
        reference: Source of these estimates.
    """

    num_qubits = attr.ib(type=int, default=0)
    duration = _PRETTY_FLOAT
    t_states_rate = _PRETTY_FLOAT
    error_rate = attr.ib(type=float, default=1e-9, repr=lambda x: f'{x:g}')
    reference = attr.ib(type=str, default='')

    def footprint(self) -> int:
        return self.num_qubits

    def n_cycles(self, n_magic: MagicStateCount) -> int:
        return n_magic.all_t_count() / self.t_states_rate * self.duration

    def distillation_error(self, n_magic: MagicStateCount, phys_err: float) -> float:
        return n_magic.all_t_count() / self.t_states_rate * phys_err
