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

from qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard import (
    _sel_hubb,
    SelectHubbard,
)
from qualtran.cirq_interop.t_complexity_protocol import t_complexity


def test_sel_hubb_auto(bloq_autotester):
    bloq_autotester(_sel_hubb)


@pytest.mark.parametrize('dim', [*range(2, 10)])
def test_select_t_complexity(dim):
    select = SelectHubbard(x_dim=dim, y_dim=dim, control_val=1)
    cost = t_complexity(select)
    N = 2 * dim * dim
    logN = 2 * (dim - 1).bit_length() + 1
    assert cost.t == 10 * N + 14 * logN - 8
    assert cost.rotations == 0
