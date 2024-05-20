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
from qualtran.bloqs.for_testing.costing import CostingBloq
from qualtran.resource_counting import get_cost_cache, get_cost_value, SuccessProb


def test_coin_flip():
    flip = CostingBloq('CoinFlip', num_qubits=1, static_costs=[(SuccessProb(), 0.5)])
    algo = CostingBloq('Algo', num_qubits=0, callees=[(flip, 4)])

    p = get_cost_value(algo, SuccessProb())
    assert p == 0.5**4

    costs = get_cost_cache(algo, SuccessProb())
    assert costs == {algo: p, flip: 0.5}
