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
from typing import Optional

from attrs import frozen

from qualtran.surface_code.data_block import DataBlock, SimpleDataBlock
from qualtran.surface_code.formulae import code_distance_from_budget, error_at
from qualtran.surface_code.magic_state_factory import MagicStateCount, MagicStateFactory
from qualtran.surface_code.physical_cost import PhysicalCost


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

    def l0_state_injection_error(self, phys_err: float) -> float:
        """Error rate associated with the level-0 creation of a |T> state.

        By using the techniques of Ying Li (https://arxiv.org/abs/1410.7808), this can be
        done with approximately the same error rate as the underlying physical error rate.
        """
        return phys_err

    def l0_topo_error_t_gate(self, phys_err: float) -> float:
        """Topological error associated with level-0 distillation.

        For a level-1 code distance of `d1`, this construction uses a `d1/2` distance code
        for storing level-0 T states.
        """

        # The chance of a logical error occurring within a lattice surgery unit cell at
        # code distance d1*0.5.
        topo_error_per_unit_cell = error_at(phys_err, d=self.distillation_l1_d // 2)

        # It takes approximately 100 L0 unit cells to get the injected state where
        # it needs to be and perform the T gate.
        return 100 * topo_error_per_unit_cell

    def l0_error(self, phys_err: float) -> float:
        """Chance of failure of a T gate performed with an injected (level-0) T state.

        As a simplifying approximation here (and elsewhere) we assume different sources
        of error are independent, and we merely add the probabilities.
        """
        return self.l0_state_injection_error(phys_err) + self.l0_topo_error_t_gate(phys_err)

    # -------------------------------------------------------------------------------
    # ----     Level 1    ---------
    # -------------------------------------------------------------------------------

    def l1_topo_error_factory(self, phys_err: float) -> float:
        """Topological error associated with a L1 T factory."""

        # The L1 T factory uses approximately 1000 L1 unit cells.
        return 1000 * error_at(phys_err, d=self.distillation_l1_d)

    def l1_topo_error_t_gate(self, phys_err: float) -> float:
        # It takes approximately 100 L1 unit cells to get the L1 state produced by the
        # factory to where it needs to be and perform the T gate.
        return 100 * error_at(phys_err, d=self.distillation_l1_d)

    def l1_distillation_error(self, phys_err: float) -> float:
        """The error due to level-0 faulty T states making it through distillation undetected.

        The level 1 distillation procedure detects any two errors. There are 35 weight-three
        errors that can make it through undetected.
        """
        return 35 * self.l0_error(phys_err) ** 3

    def l1_error(self, phys_err: float) -> float:
        """Chance of failure of a T gate performed with a T state produced from the L1 factory."""
        return (
            self.l1_topo_error_factory(phys_err)
            + self.l1_topo_error_t_gate(phys_err)
            + self.l1_distillation_error(phys_err)
        )

    # -------------------------------------------------------------------------------
    # ----     Level 2    ---------
    # -------------------------------------------------------------------------------

    def l2_error(self, phys_err: float) -> float:
        """Chance of failure of the level two factory.

        This is the chance of failure of a CCZ gate or a pair of T gates performed with a CCZ state.
        """

        # The L2 CCZ factory and catalyzed T factory both use approximately 1000 L2 unit cells.
        l2_topo_error_factory = 1000 * error_at(phys_err, d=self.distillation_l2_d)

        # Distillation error for this level.
        l2_distillation_error = 28 * self.l1_error(phys_err) ** 2

        return l2_topo_error_factory + l2_distillation_error

    # -------------------------------------------------------------------------------
    # ----     Totals    ---------
    # -------------------------------------------------------------------------------

    def footprint(self) -> int:
        l1 = 4 * 8 * 2 * self.distillation_l1_d**2
        l2 = 4 * 8 * 2 * self.distillation_l2_d**2
        return 6 * l1 + l2

    def distillation_error(self, n_magic: MagicStateCount, phys_err: float) -> float:
        """Error resulting from the magic state distillation part of the computation."""
        n_ccz_states = n_magic.ccz_count + math.ceil(n_magic.t_count / 2)
        return self.l2_error(phys_err) * n_ccz_states

    def n_cycles(self, n_magic: MagicStateCount) -> int:
        """The number of error-correction cycles to distill enough magic states."""
        distillation_d = max(2 * self.distillation_l1_d + 1, self.distillation_l2_d)
        n_ccz_states = n_magic.ccz_count + math.ceil(n_magic.t_count / 2)
        catalyzations = math.ceil(n_magic.t_count / 2)

        # Naive depth of 8.5, but can be overlapped to effective depth of 5.5
        # See section 2, paragraph 2 of the reference.
        ccz_depth = 5.5

        return math.ceil((n_ccz_states * ccz_depth + catalyzations) * distillation_d)


def get_ccz2t_costs(
    *,
    n_magic: MagicStateCount,
    n_algo_qubits: int,
    phys_err: float = 1e-3,
    error_budget: Optional[float] = 1e-2,
    cycle_time_us: float = 1.0,
    routing_overhead: Optional[float] = 0.5,
    factory: MagicStateFactory = None,
    data_block: DataBlock = None,
) -> PhysicalCost:
    """Physical costs using the model from catalyzed CCZ to 2T paper.

    Args:
        n_magic: The number of magic states (T, Toffli) required to execute the algorithm
        n_algo_qubits: Number of algorithm logical qubits.
        phys_err: The physical error rate of the device. This sets the suppression
            factor for increasing code distance.
        error_budget: The acceptable chance of an error occurring at any point. This includes
            data storage failures as well as top-level distillation failure. By default,
            this follows the prescription of the paper: distillation error is fixed by
            factory parameters and `n_magic`. The data block code distance is then chosen
            from the remaining error budget. If distillation error exceeds the budget, the cost
            estimate will fail. If the `data_block` argument is provided, this argument is
            ignored.
        cycle_time_us: The number of microseconds it takes to execute a surface code cycle.
        routing_overhead: Additional space needed for moving magic states and data qubits around
            in order to perform operations. If the `data_block` argument is provided, this
            argument is ignored.
        factory: By default, construct a default `CCZ2TFactory()`. Otherwise, you can provide
            your own factory or factory configuration using this argument.
        data_block: By default, construct a `SimpleDataBlock()` according to the `error_budget`.
            Otherwise, provide your own data block.

    References:
        Efficient magic state factories with a catalyzed |CCZ> to 2|T> transformation.
        https://arxiv.org/abs/1812.01238
    """
    if factory is None:
        factory = CCZ2TFactory()

    distillation_error = factory.distillation_error(n_magic=n_magic, phys_err=phys_err)
    n_cycles = factory.n_cycles(n_magic=n_magic)

    if data_block is None:
        # Use "left over" budget for data qubits.
        err_budget = error_budget - distillation_error
        n_logical_qubits = math.ceil((1 + routing_overhead) * n_algo_qubits)
        data_unit_cells = n_logical_qubits * n_cycles
        target_err_per_round = err_budget / data_unit_cells
        data_d = code_distance_from_budget(phys_err=phys_err, budget=target_err_per_round)
        data_block = SimpleDataBlock(data_d=data_d, routing_overhead=routing_overhead)

    data_error = data_block.data_error(
        n_algo_qubits=n_algo_qubits, n_cycles=n_cycles, phys_err=phys_err
    )
    failure_prob = distillation_error + data_error
    footprint = factory.footprint() + data_block.footprint(n_algo_qubits=n_algo_qubits)
    duration_hr = (cycle_time_us * n_cycles) / (1_000_000 * 60 * 60)

    return PhysicalCost(failure_prob=failure_prob, footprint=footprint, duration_hr=duration_hr)
