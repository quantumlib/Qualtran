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

from typing import Callable, List

import attrs

from qualtran import Bloq
from qualtran.bloqs.basic_gates import Hadamard, TGate
from qualtran.bloqs.for_testing.costing import CostingBloq, make_example_costing_bloqs
from qualtran.resource_counting import (
    CostKey,
    get_bloq_callee_counts,
    get_cost_cache,
    get_cost_value,
)
from qualtran.resource_counting.generalizers import generalize_rotation_angle


class TestCostKey(CostKey[int]):
    def __init__(self):
        # For testing, keep a log of all the bloqs for which we called 'compute' on.
        self._log: List[Bloq] = []

    def compute(self, bloq: 'Bloq', get_callee_cost: Callable[['Bloq'], int]) -> int:
        self._log.append(bloq)

        total = 1
        for callee, n_times_called in get_bloq_callee_counts(bloq):
            total += n_times_called * get_callee_cost(callee)

        return total

    def zero(self) -> int:
        return 0

    def __hash__(self):
        return hash(self.__class__)

    def __eq__(self, other):
        return isinstance(other, self.__class__)


def test_get_cost_value_caching():
    cost = TestCostKey()
    algo = make_example_costing_bloqs()
    assert isinstance(algo, CostingBloq)
    _ = get_cost_value(algo, cost)
    n_times_compute_called_on_t = sum(b == TGate() for b in cost._log)
    assert n_times_compute_called_on_t == 1, 'should use cached value'


def test_get_cost_value_static():
    algo = make_example_costing_bloqs()

    # Modify "func1" to have static costs
    func1 = algo.callees[0][0]
    func1_mod = attrs.evolve(func1, static_costs=[(TestCostKey(), 123)])
    algo_mod = attrs.evolve(algo, callees=[(func1_mod, 1), algo.callees[1]])
    assert get_cost_value(func1_mod, TestCostKey()) == 123

    # Should not call "compute" for Func1, since it has static costs
    # Should not have to recurse into H, T^dag; which is only used by Func1
    cost = TestCostKey()
    _ = get_cost_value(algo_mod, cost)
    assert len(cost._log) == 3
    assert 'Func2' in [str(b) for b in cost._log]
    assert 'Func1' not in [str(b) for b in cost._log]
    assert TGate().adjoint() not in cost._log
    assert Hadamard() not in cost._log


def test_get_cost_value_static_user_provided():
    cost = TestCostKey()
    algo = make_example_costing_bloqs()

    # Provide cached costs up front for func1
    func1 = algo.callees[0][0]

    # Should not call "compute" for Func1, since we supplied an existing cache
    # Should not have to recurse into H, T^dag; which is only used by Func1
    _ = get_cost_value(algo, cost, costs_cache={func1: 0})
    assert len(cost._log) == 3
    assert 'Func2' in [str(b) for b in cost._log]
    assert 'Func1' not in [str(b) for b in cost._log]
    assert TGate().adjoint() not in cost._log
    assert Hadamard() not in cost._log


def test_costs_generalizer():
    assert generalize_rotation_angle(TGate().adjoint()) == TGate()

    algo = CostingBloq(name='algo', num_qubits=1, callees=[(TGate(), 1), (TGate().adjoint(), 1)])
    cost_cache = get_cost_cache(algo, TestCostKey())
    assert cost_cache[algo] == 3

    cost_cache_gen = get_cost_cache(algo, TestCostKey(), generalizer=generalize_rotation_angle)
    assert TGate() in cost_cache_gen
    assert TGate().adjoint() not in cost_cache_gen
    assert cost_cache_gen[algo] == 3
