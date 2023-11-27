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

from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.chemistry.pbc.first_quantization.projectile.prepare_nu import (
    _prep_mu_proj,
    _prep_nu_proj,
    PrepareNuStateWithProj,
)


def test_prepare_num(bloq_autotester):
    bloq_autotester(_prep_nu_proj)


def test_prepare_mu(bloq_autotester):
    bloq_autotester(_prep_mu_proj)


def test_prepare_nu_with_proj_t_counts():
    num_bits_p = 6
    num_bits_n = 8
    m_param = 2 ** (2 * num_bits_n + 3)
    num_bits_m = (m_param - 1).bit_length()
    # arithmetic + inequality + 3 Toffolis for flag (eq 89 + sentence immediately following it)
    expected_cost = 3 * num_bits_n**2 + num_bits_n + 4 * num_bits_m * (num_bits_n + 1) + 4
    # factor of two for inverserse prepare, controlled hadamard + testing on nu.
    expected_cost += (
        2 * 4 * (num_bits_n - 1) + (num_bits_n - num_bits_p - 1) + 6 * num_bits_n + 2 + 2
    )
    eq_c6 = (
        3 * num_bits_n**2 + 16 * num_bits_n - num_bits_p - 6 + 4 * num_bits_m * (num_bits_n + 1)
    )
    assert expected_cost == eq_c6 + 5
    prep = PrepareNuStateWithProj(num_bits_p, num_bits_n, m_param)
    _, counts = prep.call_graph()
    qual_cost = counts[TGate()]
    prep = PrepareNuStateWithProj(num_bits_p, num_bits_n, m_param, adjoint=True)
    _, counts = prep.call_graph()
    qual_cost += counts[TGate()]
    qual_cost //= 4
    comp_diff = 1
    assert qual_cost == expected_cost - comp_diff
