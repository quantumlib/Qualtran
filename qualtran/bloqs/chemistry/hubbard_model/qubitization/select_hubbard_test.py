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
from unittest.mock import ANY

import pytest

from qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard import (
    _sel_hubb,
    SelectHubbard,
)
from qualtran.resource_counting import GateCounts, get_cost_value, QECGatesCost


def test_sel_hubb_auto(bloq_autotester):
    bloq_autotester(_sel_hubb)


@pytest.mark.parametrize('dim', [*range(2, 10)])
def test_select_t_complexity(dim):
    select = SelectHubbard(x_dim=dim, y_dim=dim, control_val=1)
    cost = get_cost_value(select, QECGatesCost())
    N = 2 * dim * dim
    logN = 2 * (dim - 1).bit_length() + 1
    assert cost == GateCounts(
        cswap=2 * logN, and_bloq=5 * (N // 2) - 2, measurement=5 * (N // 2) - 2, clifford=ANY
    )
    assert cost.total_t_count() == 10 * N + 8 * logN - 8


def test_adjoint_controlled():
    bloq = _sel_hubb()

    adj_ctrl_bloq = bloq.controlled().adjoint()
    ctrl_adj_bloq = bloq.adjoint().controlled()

    assert adj_ctrl_bloq == ctrl_adj_bloq
