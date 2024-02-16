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
from typing import Callable, Iterable, Iterator, Optional, Tuple

from attrs import frozen

import qualtran.surface_code.quantum_error_correction_scheme_summary as qec
from qualtran.surface_code.algorithm_summary import AlgorithmSummary
from qualtran.surface_code.data_block import DataBlock, SimpleDataBlock
from qualtran.surface_code.magic_state_factory import MagicStateFactory
from qualtran.surface_code.multi_factory import MultiFactory
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
    qec_scheme: qec.QuantumErrorCorrectionSchemeSummary = qec.FowlerSuperconductingQubits

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
        topo_error_per_unit_cell = self.qec_scheme.logical_error_rate(
            physical_error_rate=phys_err, code_distance=self.distillation_l1_d // 2
        )

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
        return 1000 * self.qec_scheme.logical_error_rate(
            physical_error_rate=phys_err, code_distance=self.distillation_l1_d
        )

    def l1_topo_error_t_gate(self, phys_err: float) -> float:
        # It takes approximately 100 L1 unit cells to get the L1 state produced by the
        # factory to where it needs to be and perform the T gate.
        return 100 * self.qec_scheme.logical_error_rate(
            physical_error_rate=phys_err, code_distance=self.distillation_l1_d
        )

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
        l2_topo_error_factory = 1000 * self.qec_scheme.logical_error_rate(
            physical_error_rate=phys_err, code_distance=self.distillation_l2_d
        )

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

    def distillation_error(self, n_magic: AlgorithmSummary, phys_err: float) -> float:
        """Error resulting from the magic state distillation part of the computation."""
        n_ccz_states = n_magic.toffoli_gates + math.ceil(n_magic.t_gates / 2)
        return self.l2_error(phys_err) * n_ccz_states

    def n_cycles(self, n_magic: AlgorithmSummary) -> int:
        """The number of error-correction cycles to distill enough magic states."""
        distillation_d = max(2 * self.distillation_l1_d + 1, self.distillation_l2_d)
        n_ccz_states = n_magic.toffoli_gates + math.ceil(n_magic.t_gates / 2)
        catalyzations = math.ceil(n_magic.t_gates / 2)

        # Naive depth of 8.5, but can be overlapped to effective depth of 5.5
        # See section 2, paragraph 2 of the reference.
        ccz_depth = 5.5

        return math.ceil((n_ccz_states * ccz_depth + catalyzations) * distillation_d)


def get_ccz2t_costs(
    *,
    n_magic: AlgorithmSummary,
    n_algo_qubits: int,
    phys_err: float,
    cycle_time_us: float,
    factory: MagicStateFactory,
    data_block: DataBlock,
) -> PhysicalCost:
    """Generate spacetime cost and failure probability given physical and logical parameters.

    Note that this function can return failure probabilities larger than 1.

    Args:
        n_magic: The number of magic states (T, Toffoli) required to execute the algorithm
        n_algo_qubits: Number of algorithm logical qubits.
        phys_err: The physical error rate of the device.
        cycle_time_us: The number of microseconds it takes to execute a surface code cycle.
        factory: magic state factory configuration. Used to evaluate distillation error and cost.
        data_block: data block configuration. Used to evaluate data error and footprint.
    """
    distillation_error = factory.distillation_error(n_magic=n_magic, phys_err=phys_err)
    n_cycles = factory.n_cycles(n_magic=n_magic)
    data_error = data_block.data_error(
        n_algo_qubits=n_algo_qubits, n_cycles=n_cycles, phys_err=phys_err
    )
    failure_prob = distillation_error + data_error
    footprint = factory.footprint() + data_block.footprint(n_algo_qubits=n_algo_qubits)
    duration_hr = (cycle_time_us * n_cycles) / (1_000_000 * 60 * 60)

    return PhysicalCost(failure_prob=failure_prob, footprint=footprint, duration_hr=duration_hr)


def get_ccz2t_costs_from_error_budget(
    *,
    n_magic: AlgorithmSummary,
    n_algo_qubits: int,
    phys_err: float = 1e-3,
    error_budget: float = 1e-2,
    cycle_time_us: float = 1.0,
    routing_overhead: float = 0.5,
    factory: Optional[MagicStateFactory] = None,
    data_block: Optional[DataBlock] = None,
) -> PhysicalCost:
    """Physical costs using the model from catalyzed CCZ to 2T paper.

    Args:
        n_magic: The number of magic states (T, Toffoli) required to execute the algorithm
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
        if err_budget < 0:
            raise ValueError(
                f"distillation error {distillation_error} is larger than the error budget {error_budget}"
            )
        n_logical_qubits = math.ceil((1 + routing_overhead) * n_algo_qubits)
        data_unit_cells = n_logical_qubits * n_cycles
        target_err_per_round = err_budget / data_unit_cells
        data_d = qec.FowlerSuperconductingQubits.code_distance_from_budget(
            physical_error_rate=phys_err, budget=target_err_per_round
        )
        data_block = SimpleDataBlock(
            data_d=data_d,
            routing_overhead=routing_overhead,
            qec_scheme=qec.FowlerSuperconductingQubits,
        )

    return get_ccz2t_costs(
        n_magic=n_magic,
        n_algo_qubits=n_algo_qubits,
        phys_err=phys_err,
        cycle_time_us=cycle_time_us,
        factory=factory,
        data_block=data_block,
    )


def iter_ccz2t_factories(
    l1_start: int = 5, l1_stop: int = 25, l2_stop: int = 41, *, n_factories=1
) -> Iterator[MagicStateFactory]:
    """Iterate over CCZ2T (multi)factories in the given range of distillation code distances

    Args:
        l1_start (int, optional): Minimum level 1 distillation distance.
        l1_stop (int, optional): Maximum level 1 distillation distance.
        l2_stop (int, optional): Maximum level 2 distillation distance. The minimum is
            automatically chosen as 2 + l1_distance, ensuring l2_distance > l1_distance.
        n_factories (int, optional): Number of factories to be used in parallel.
    """
    if n_factories == 1:
        factory = CCZ2TFactory
    elif n_factories > 1:

        def factory(distillation_l1_d, distillation_l2_d):
            base_factory = CCZ2TFactory(
                distillation_l1_d=l1_distance, distillation_l2_d=l2_distance
            )
            return MultiFactory(base_factory=base_factory, n_factories=n_factories)

    else:
        raise ValueError("The number of factories should be a positive integer")

    for l1_distance in range(l1_start, l1_stop, 2):
        for l2_distance in range(l1_distance + 2, l2_stop, 2):
            yield factory(distillation_l1_d=l1_distance, distillation_l2_d=l2_distance)


def iter_simple_data_blocks(d_start: int = 7, d_stop: int = 35):
    for logical_data_qubit_distance in range(d_start, d_stop, 2):
        yield SimpleDataBlock(data_d=logical_data_qubit_distance)


def get_ccz2t_costs_from_grid_search(
    *,
    n_magic: AlgorithmSummary,
    n_algo_qubits: int,
    phys_err: float = 1e-3,
    error_budget: float = 1e-2,
    cycle_time_us: float = 1.0,
    factory_iter: Iterable[MagicStateFactory] = tuple(iter_ccz2t_factories()),
    data_block_iter: Iterable[DataBlock] = tuple(iter_simple_data_blocks()),
    cost_function: Callable[[PhysicalCost], float] = (lambda pc: pc.qubit_hours),
) -> Tuple[PhysicalCost, CCZ2TFactory, SimpleDataBlock]:
    """Grid search over parameters to minimize space time volume.

    Args:
        n_magic: The number of magic states (T, Toffoli) required to execute the algorithm
        n_algo_qubits: Number of algorithm logical qubits.
        phys_err: The physical error rate of the device. This sets the suppression
            factor for increasing code distance.
        error_budget: The acceptable chance of an error occurring at any point. This includes
            data storage failures as well as top-level distillation failure.
        cycle_time_us: The number of microseconds it takes to execute a surface code cycle.
        factory_iter: iterable containing the instances of MagicStateFactory to search over.
        data_block_iter: iterable containing the instances of SimpleDataBlock to search over.
        cost_function: function of PhysicalCost to be minimized. Defaults to spacetime volume.
            Set `cost_function = (lambda pc: pc.duration_hr)` to mimimize wall time.

    Returns:
        best_cost, best_factory, best_data_block

    References:
        A similar search was conducted manually in https://arxiv.org/abs/2011.03494, using a tweaked
        version of the spreadsheet from https://arxiv.org/abs/1812.01238
    """
    best_cost: Optional[PhysicalCost] = None
    best_params: Optional[Tuple[CCZ2TFactory, SimpleDataBlock]] = None
    for factory in factory_iter:
        for data_block in data_block_iter:
            cost = get_ccz2t_costs(
                n_magic=n_magic,
                n_algo_qubits=n_algo_qubits,
                phys_err=phys_err,
                cycle_time_us=cycle_time_us,
                factory=factory,
                data_block=data_block,
            )

            if cost.failure_prob > error_budget:
                continue
            if best_cost is None or cost_function(cost) < cost_function(best_cost):
                best_cost = cost
                best_params = (factory, data_block)

    best_factory, best_data_block = best_params
    return best_cost, best_factory, best_data_block
