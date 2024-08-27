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

from qualtran.bloqs.chemistry.pbc.first_quantization.projectile.prepare_uv import (
    _prep_uv_proj,
    PrepareUVFirstQuantizationWithProj,
)
from qualtran.resource_counting import get_cost_value, QECGatesCost


def test_prep_uv_proj(bloq_autotester):
    bloq_autotester(_prep_uv_proj)


def test_prepare_uv_t_counts():
    num_bits_p = 6
    num_bits_n = 8
    eta = 10
    num_atoms = 10
    lambda_zeta = 10
    num_bits_nuc_pos = 8
    m_param = 2 ** (2 * num_bits_n + 3)
    num_bits_m = (m_param - 1).bit_length()
    expected_cost = 3 * num_bits_n**2 + num_bits_n + 4 * num_bits_m * (num_bits_n + 1) + 4  # C4
    expected_cost += (
        2 * 4 * (num_bits_n - 1) + (num_bits_n - num_bits_p - 1) + 6 * num_bits_n + 2 + 2
    )  # C4
    expected_cost += lambda_zeta + int(np.ceil(lambda_zeta**0.5))
    prep = PrepareUVFirstQuantizationWithProj(
        num_bits_p, num_bits_n, eta, num_atoms, m_param, lambda_zeta, num_bits_nuc_pos
    )
    qual_cost = get_cost_value(prep, QECGatesCost()).total_t_count()
    prep = PrepareUVFirstQuantizationWithProj(
        num_bits_p, num_bits_n, eta, num_atoms, m_param, lambda_zeta, num_bits_nuc_pos
    ).adjoint()
    qual_cost += get_cost_value(prep, QECGatesCost()).total_t_count()
    qual_cost //= 4
    comp_diff = 1
    assert qual_cost == expected_cost - comp_diff
