#  Copyright 2024 Google LLC
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

from .magic_state_factory import MagicStateFactory

if TYPE_CHECKING:
    from qualtran.resource_counting import GateCounts
    from qualtran.surface_code import LogicalErrorModel


@frozen
class CCZ2TFactory(MagicStateFactory):
    """Magic state factory costs using the model from catalyzed CCZ to 2T paper.

    Args:
        distillation_l1_d: Code distance used for level 1 factories.
        distillation_l2_d: Code distance used for level 2 factories.

    References:
        Efficient magic state factories with a catalyzed |CCZ> to 2|T> transformation.
        https://arxiv.org/abs/1812.01238
    """

    distillation_l1_d: int = 15
    distillation_l2_d: int = 31

    # -------------------------------------------------------------------------------
    # ----     Level 0    ---------
    # -------------------------------------------------------------------------------

    def l0_state_injection_error(self, error_model: 'LogicalErrorModel') -> float:
        """Error rate associated with the level-0 creation of a |T> state.

        By using the techniques of Ying Li (https://arxiv.org/abs/1410.7808), this can be
        done with approximately the same error rate as the underlying physical error rate.
        """
        return error_model.physical_error

    def l0_topo_error_t_gate(self, error_model: 'LogicalErrorModel') -> float:
        """Topological error associated with level-0 distillation.

        For a level-1 code distance of `d1`, this construction uses a `d1/2` distance code
        for storing level-0 T states.
        """

        # The chance of a logical error occurring within a lattice surgery unit cell at
        # code distance d1*0.5.
        topo_error_per_unit_cell = error_model(code_distance=self.distillation_l1_d // 2)

        # It takes approximately 100 L0 unit cells to get the injected state where
        # it needs to be and perform the T gate.
        return 100 * topo_error_per_unit_cell

    def l0_error(self, error_model: 'LogicalErrorModel') -> float:
        """Chance of failure of a T gate performed with an injected (level-0) T state.

        As a simplifying approximation here (and elsewhere) we assume different sources
        of error are independent, and we merely add the probabilities.
        """
        return self.l0_state_injection_error(error_model) + self.l0_topo_error_t_gate(error_model)

    # -------------------------------------------------------------------------------
    # ----     Level 1    ---------
    # -------------------------------------------------------------------------------

    def l1_topo_error_factory(self, error_model: 'LogicalErrorModel') -> float:
        """Topological error associated with a L1 T factory."""

        # The L1 T factory uses approximately 1000 L1 unit cells.
        return 1000 * error_model(code_distance=self.distillation_l1_d)

    def l1_topo_error_t_gate(self, error_model: 'LogicalErrorModel') -> float:
        # It takes approximately 100 L1 unit cells to get the L1 state produced by the
        # factory to where it needs to be and perform the T gate.
        return 100 * error_model(code_distance=self.distillation_l1_d)

    def l1_distillation_error(self, error_model: 'LogicalErrorModel') -> float:
        """The error due to level-0 faulty T states making it through distillation undetected.

        The level 1 distillation procedure detects any two errors. There are 35 weight-three
        errors that can make it through undetected.
        """
        return 35 * self.l0_error(error_model) ** 3

    def l1_error(self, error_model: 'LogicalErrorModel') -> float:
        """Chance of failure of a T gate performed with a T state produced from the L1 factory."""
        return (
            self.l1_topo_error_factory(error_model)
            + self.l1_topo_error_t_gate(error_model)
            + self.l1_distillation_error(error_model)
        )

    # -------------------------------------------------------------------------------
    # ----     Level 2    ---------
    # -------------------------------------------------------------------------------

    def l2_error(self, error_model: 'LogicalErrorModel') -> float:
        """Chance of failure of the level two factory.

        This is the chance of failure of a CCZ gate or a pair of T gates performed with a CCZ state.
        """

        # The L2 CCZ factory and catalyzed T factory both use approximately 1000 L2 unit cells.
        l2_topo_error_factory = 1000 * error_model(self.distillation_l2_d)

        # Distillation error for this level.
        l2_distillation_error = 28 * self.l1_error(error_model) ** 2

        return l2_topo_error_factory + l2_distillation_error

    # -------------------------------------------------------------------------------
    # ----     Totals    ---------
    # -------------------------------------------------------------------------------

    def n_physical_qubits(self) -> int:
        l1 = 4 * 8 * 2 * self.distillation_l1_d**2
        l2 = 4 * 8 * 2 * self.distillation_l2_d**2
        return 6 * l1 + l2

    def factory_error(
        self, n_logical_gates: 'GateCounts', logical_error_model: 'LogicalErrorModel'
    ) -> float:
        """Error resulting from the magic state distillation part of the computation."""
        counts = n_logical_gates.total_t_and_ccz_count()
        total_ccz_states = counts['n_ccz'] + math.ceil(counts['n_t'] / 2)
        return self.l2_error(logical_error_model) * total_ccz_states

    def n_cycles(
        self, n_logical_gates: 'GateCounts', logical_error_model: 'LogicalErrorModel'
    ) -> int:
        """The number of error-correction cycles to distill enough magic states."""
        distillation_d = max(2 * self.distillation_l1_d + 1, self.distillation_l2_d)
        counts = n_logical_gates.total_t_and_ccz_count()
        n_ccz_states = counts['n_ccz'] + math.ceil(counts['n_t'] / 2)
        catalyzations = math.ceil(counts['n_t'] / 2)

        # Naive depth of 8.5, but can be overlapped to effective depth of 5.5
        # See section 2, paragraph 2 of the reference.
        ccz_depth = 5.5

        return math.ceil((n_ccz_states * ccz_depth + catalyzations) * distillation_d)
