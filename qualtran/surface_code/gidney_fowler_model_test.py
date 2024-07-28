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

from qualtran.resource_counting import GateCounts
from qualtran.surface_code import (
    CCZ2TFactory,
    get_ccz2t_costs_from_error_budget,
    get_ccz2t_costs_from_grid_search,
    iter_ccz2t_factories,
    MultiFactory,
)


def test_vs_spreadsheet():
    re = get_ccz2t_costs_from_error_budget(
        n_logical_gates=GateCounts(t=10**8, toffoli=10**8),
        n_algo_qubits=100,
        error_budget=0.01,
        phys_err=1e-3,
        cycle_time_us=1,
    )

    np.testing.assert_allclose(re.failure_prob, 0.0084, rtol=1e-3)
    np.testing.assert_allclose(re.footprint, 4.00e5, rtol=1e-3)
    np.testing.assert_allclose(re.duration_hr, 7.53, rtol=1e-3)


def test_grid_search_runs():
    cost, factory, db = get_ccz2t_costs_from_grid_search(
        n_logical_gates=GateCounts(t=10**8, toffoli=10**8),
        n_algo_qubits=100,
        phys_err=1e-3,
        error_budget=0.1,
        cycle_time_us=1,
    )
    assert isinstance(factory, CCZ2TFactory)
    assert factory.distillation_l1_d == 15
    assert factory.distillation_l2_d == 23
    assert db.data_d == 25


def test_grid_search_against_thc():
    """test based on the parameters reported in section IV.C of Lee et al., PRXQuantum 2, 2021"""
    best_cost, best_factory, best_data_block = get_ccz2t_costs_from_grid_search(
        n_logical_gates=GateCounts(toffoli=6665400000),
        n_algo_qubits=696,
        error_budget=1e-2,
        phys_err=1e-3,
        factory_iter=iter_ccz2t_factories(n_factories=4),
        cost_function=(lambda pc: pc.duration_hr),
    )
    assert best_cost.failure_prob == 0.007725395132201774
    assert best_cost.footprint == 2933032
    assert best_cost.duration_hr == 89.1034375
    assert isinstance(best_factory, MultiFactory)
    base_factory = best_factory.base_factory
    assert isinstance(base_factory, CCZ2TFactory)
    assert base_factory.distillation_l1_d == 17
    assert base_factory.distillation_l2_d == 29
    assert best_data_block.data_d == 33
