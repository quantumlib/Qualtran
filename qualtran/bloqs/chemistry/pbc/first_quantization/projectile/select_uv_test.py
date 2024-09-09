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
from qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_uv import (
    _sel_uv_proj,
    SelectUVFirstQuantizationWithProj,
)
from qualtran.resource_counting import get_cost_value, QECGatesCost


def test_sel_uv_proj(bloq_autotester):
    bloq_autotester(_sel_uv_proj)


def test_select_uv_toffoli_counts():
    num_bits_p = 6
    num_bits_n = 9
    eta = 10
    num_bits_nuc_pos = 18
    expected_cost = 3 * (num_bits_p - 2) + 3 * (num_bits_n - 2)
    expected_cost += 6 * (num_bits_p + 1) + 6 * (num_bits_n + 1)
    expected_cost += 3 * num_bits_p + 3 * num_bits_n
    expected_cost += 3 * (2 * num_bits_n * num_bits_nuc_pos - num_bits_n * (num_bits_n + 1) - 1)
    sel = SelectUVFirstQuantizationWithProj(num_bits_p, num_bits_n, eta, eta, num_bits_nuc_pos)
    qual_cost = get_cost_value(sel, QECGatesCost()).total_toffoli_only()
    # -6 due to different cost of addition.
    qual_cost += 6
    assert qual_cost == expected_cost
