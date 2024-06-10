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

from qualtran.bloqs.chemistry.hubbard_model.qubitization.prepare_hubbard import (
    _prep_hubb,
    PrepareHubbard,
)
from qualtran.cirq_interop.t_complexity_protocol import t_complexity


def test_prep_hubb_auto(bloq_autotester):
    bloq_autotester(_prep_hubb)


@pytest.mark.parametrize('dim', [*range(3, 10)])
def test_prepare_t_complexity(dim):
    prepare = PrepareHubbard(x_dim=dim, y_dim=dim, t=2, u=8)
    cost = t_complexity(prepare)
    logN = 2 * (dim - 1).bit_length() + 1
    assert cost.t <= 32 * logN
    # TODO(#233): The rotation count should reduce to a constant once cost for Controlled-H
    # gates is recognized as $2$ T-gates instead of $2$ rotations.
    assert cost.rotations <= 2 * logN + 9
