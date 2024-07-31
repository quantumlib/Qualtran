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
from typing import Callable, cast, Iterable, Iterator, Optional, Tuple, TYPE_CHECKING

from .algorithm_summary import AlgorithmSummary
from .ccz2t_factory import CCZ2TFactory
from .data_block import DataBlock, SimpleDataBlock
from .magic_state_factory import MagicStateFactory
from .multi_factory import MultiFactory
from .physical_cost_model import PhysicalCostModel
from .physical_cost_summary import PhysicalCostsSummary
from .physical_parameters import PhysicalParameters
from .qec_scheme import QECScheme

if TYPE_CHECKING:
    from qualtran.resource_counting import GateCounts


def get_ccz2t_costs(
    *,
    n_logical_gates: 'GateCounts',
    n_algo_qubits: int,
    phys_err: float,
    cycle_time_us: float,
    factory: MagicStateFactory,
    data_block: DataBlock,
) -> PhysicalCostsSummary:
    """Generate spacetime cost and failure probability given physical and logical parameters.

    Note that this function can return failure probabilities larger than 1.

    This function exists for backwards-compatibility. Consider constructing a `PhysicalCostModel`
    directly.

    Args:
        n_logical_gates: The number of algorithm logical gates.
        n_algo_qubits: Number of algorithm logical qubits.
        phys_err: The physical error rate of the device.
        cycle_time_us: The number of microseconds it takes to execute a surface code cycle.
        factory: magic state factory configuration. Used to evaluate distillation error and cost.
        data_block: data block configuration. Used to evaluate data error and footprint.
    """

    model = PhysicalCostModel(
        physical_params=PhysicalParameters(physical_error=phys_err, cycle_time_us=cycle_time_us),
        data_block=data_block,
        factory=factory,
        qec_scheme=QECScheme.make_gidney_fowler(),
    )
    algo = AlgorithmSummary(n_algo_qubits=n_algo_qubits, n_logical_gates=n_logical_gates)
    return PhysicalCostsSummary(
        failure_prob=model.error(algo),
        footprint=model.n_phys_qubits(algo),
        duration_hr=model.duration_hr(algo),
    )


def get_ccz2t_costs_from_error_budget(
    *,
    n_logical_gates: 'GateCounts',
    n_algo_qubits: int,
    phys_err: float = 1e-3,
    error_budget: float = 1e-2,
    cycle_time_us: float = 1.0,
    routing_overhead: float = 0.5,
    factory: Optional[MagicStateFactory] = None,
    data_block: Optional[DataBlock] = None,
) -> PhysicalCostsSummary:
    """Physical costs using the model from catalyzed CCZ to 2T paper.

    Args:
        n_logical_gates: Number of algorithm logical gates.
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

    from qualtran.surface_code import LogicalErrorModel, QECScheme

    qec_scheme = QECScheme.make_gidney_fowler()
    err_model = LogicalErrorModel(qec_scheme=qec_scheme, physical_error=phys_err)
    factory_error = factory.factory_error(
        n_logical_gates=n_logical_gates, logical_error_model=err_model
    )
    n_cycles = factory.n_cycles(n_logical_gates=n_logical_gates, logical_error_model=err_model)

    if data_block is None:
        # Use "left over" budget for data qubits.
        err_budget = error_budget - factory_error
        if err_budget < 0:
            raise ValueError(
                f"Factory error {factory_error} is larger than the error budget {error_budget}"
            )
        n_logical_qubits = math.ceil((1 + routing_overhead) * n_algo_qubits)
        data_unit_cells = n_logical_qubits * n_cycles
        target_err_per_round = err_budget / data_unit_cells
        data_d = qec_scheme.code_distance_from_budget(
            physical_error=phys_err, budget=target_err_per_round
        )
        data_block = SimpleDataBlock(data_d=data_d, routing_overhead=routing_overhead)

    return get_ccz2t_costs(
        n_logical_gates=n_logical_gates,
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
    factory: Callable[[int, int], MagicStateFactory]
    if n_factories == 1:
        factory = CCZ2TFactory
    elif n_factories > 1:

        def factory(distillation_l1_d: int, distillation_l2_d: int) -> MagicStateFactory:
            base_factory = CCZ2TFactory(
                distillation_l1_d=distillation_l1_d, distillation_l2_d=distillation_l2_d
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
    n_logical_gates: 'GateCounts',
    n_algo_qubits: int,
    phys_err: float = 1e-3,
    error_budget: float = 1e-2,
    cycle_time_us: float = 1.0,
    factory_iter: Iterable[MagicStateFactory] = tuple(iter_ccz2t_factories()),
    data_block_iter: Iterable[DataBlock] = tuple(iter_simple_data_blocks()),
    cost_function: Callable[[PhysicalCostsSummary], float] = (lambda pc: pc.qubit_hours),
) -> Tuple[PhysicalCostsSummary, MagicStateFactory, SimpleDataBlock]:
    """Grid search over parameters to minimize the space-time volume.

    Args:
        n_logical_gates: Number of algorithm logical gates.
        n_algo_qubits: Number of algorithm logical qubits.
        phys_err: The physical error rate of the device. This sets the suppression
            factor for increasing code distance.
        error_budget: The acceptable chance of an error occurring at any point. This includes
            data storage failures as well as top-level distillation failure.
        cycle_time_us: The number of microseconds it takes to execute a surface code cycle.
        factory_iter: iterable containing the instances of MagicStateFactory to search over.
        data_block_iter: iterable containing the instances of SimpleDataBlock to search over.
        cost_function: function of PhysicalCostsSummary to be minimized. Defaults to spacetime volume.
            Set `cost_function = (lambda pc: pc.duration_hr)` to mimimize wall time.

    Returns:
        best_cost, best_factory, best_data_block

    References:
        A similar search was conducted manually in https://arxiv.org/abs/2011.03494, using a tweaked
        version of the spreadsheet from https://arxiv.org/abs/1812.01238
    """
    best_cost: Optional[PhysicalCostsSummary] = None
    best_params: Optional[Tuple[MagicStateFactory, SimpleDataBlock]] = None
    for factory in factory_iter:
        for data_block in data_block_iter:
            cost = get_ccz2t_costs(
                n_logical_gates=n_logical_gates,
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
                best_params = (factory, cast(SimpleDataBlock, data_block))

    if best_params is None or best_cost is None:
        raise ValueError("No valid factories found!")

    best_factory, best_data_block = best_params
    return best_cost, best_factory, best_data_block
