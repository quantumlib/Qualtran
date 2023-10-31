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

import pytest

import qualtran.surface_code.rotation_cost_model as rcm
from qualtran.surface_code.algorithm_summary import AlgorithmSummary


@pytest.mark.parametrize(
    'model,want',
    [
        (rcm.BeverlandEtAlRotationCost, AlgorithmSummary(t_gates=7)),
        (
            rcm.ConstantWithOverheadRotationCost(
                bitsize=13, overhead_rotation_cost=rcm.RotationLogarithmicModel(1, 1)
            ),
            AlgorithmSummary(toffoli_gates=11),
        ),
    ],
)
def test_rotation_cost(model: rcm.RotationCostModel, want: float):
    assert model.rotation_cost(2**-3) == want


@pytest.mark.parametrize(
    'model,want',
    [
        (rcm.BeverlandEtAlRotationCost, AlgorithmSummary()),
        (
            rcm.ConstantWithOverheadRotationCost(
                bitsize=13, overhead_rotation_cost=rcm.RotationLogarithmicModel(1, 1)
            ),
            AlgorithmSummary(t_gates=104),
        ),
    ],
)
def test_preparation_overhead(model: rcm.RotationCostModel, want: float):
    assert model.prepartion_overhead(2**-3) == want
