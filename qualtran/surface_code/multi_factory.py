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

import numpy as np
from attrs import frozen

from qualtran.surface_code.algorithm_summary import AlgorithmSummary
from qualtran.surface_code.magic_state_factory import MagicStateFactory


@frozen
class MultiFactory(MagicStateFactory):
    """Overlay of MagicStateFactory representing multiple factories of the same kind.

    All quantities are derived by those of `base_factory`. `footprint` is multiplied by
    `n_factories`, `n_cycles` is divided by `n_factoties`, and  `distillation_error` is independent
    on the number of factories.

    Args:
        base_factory: the base factory to be replicated.
        n_factories: number of factories to construct.
    """

    base_factory: MagicStateFactory
    n_factories: int

    def footprint(self) -> int:
        return self.base_factory.footprint() * self.n_factories

    def n_cycles(self, n_magic: AlgorithmSummary) -> int:
        return np.ceil(self.base_factory.n_cycles(n_magic) / self.n_factories)

    def distillation_error(self, n_magic: AlgorithmSummary, phys_err: float) -> float:
        return self.base_factory.distillation_error(n_magic, phys_err)
