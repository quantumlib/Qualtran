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
from typing import TYPE_CHECKING

from attrs import frozen

from qualtran.surface_code.magic_state_factory import MagicStateFactory

if TYPE_CHECKING:
    from qualtran.resource_counting import GateCounts
    from qualtran.surface_code import LogicalErrorModel


@frozen
class MultiFactory(MagicStateFactory):
    """Overlay of MagicStateFactory representing multiple factories of the same kind.

    All quantities are derived by those of `base_factory`. `footprint` is multiplied by
    `n_factories`, `n_cycles` is divided by `n_factories`, and  `distillation_error` is independent
    on the number of factories.

    Args:
        base_factory: the base factory to be replicated.
        n_factories: number of factories to construct.
    """

    base_factory: MagicStateFactory
    n_factories: int

    def n_physical_qubits(self) -> int:
        return self.base_factory.n_physical_qubits() * self.n_factories

    def n_cycles(
        self, n_logical_gates: 'GateCounts', logical_error_model: 'LogicalErrorModel'
    ) -> int:
        return math.ceil(
            self.base_factory.n_cycles(n_logical_gates, logical_error_model) / self.n_factories
        )

    def factory_error(
        self, n_logical_gates: 'GateCounts', logical_error_model: 'LogicalErrorModel'
    ) -> float:
        return self.base_factory.factory_error(n_logical_gates, logical_error_model)
