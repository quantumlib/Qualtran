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

from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.chemistry.pbc.first_quantization.projectile.prepare_t import (
    _prep_power_two_proj,
    _prep_t_proj,
    PreparePowerTwoStateWithProj,
    PrepareTFirstQuantizationWithProj,
)


def test_prep_t_proj(bloq_autotester):
    bloq_autotester(_prep_t_proj)


def test_prep_power_two_proj(bloq_autotester):
    bloq_autotester(_prep_power_two_proj)


def test_prepare_kinetic_t_proj_counts():
    num_bits_p = 6
    num_bits_n = 8
    eta = 10
    b_r = 8
    expected_cost = 2 * (2 * num_bits_n + 9) + 2 * (num_bits_n - num_bits_p) + 20
    qual_cost = 0
    prep = PrepareTFirstQuantizationWithProj(num_bits_p, num_bits_n, eta, num_bits_rot_aa=b_r)
    _, counts = prep.call_graph()
    qual_cost += counts[TGate()]
    prep = PrepareTFirstQuantizationWithProj(
        num_bits_p, num_bits_n, eta, num_bits_rot_aa=b_r, adjoint=True
    )
    _, counts = prep.call_graph()
    qual_cost += counts[TGate()]
    qual_cost //= 4
    assert qual_cost == expected_cost


def test_t_proj_attrs_post_init():
    with pytest.raises(ValueError):
        PreparePowerTwoStateWithProj(6, 8)
