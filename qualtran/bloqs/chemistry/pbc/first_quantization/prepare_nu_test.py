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

from qualtran.bloqs.chemistry.pbc.first_quantization.prepare_nu import PrepareNuState
from qualtran.resource_counting import get_cost_value, QECGatesCost
from qualtran.testing import assert_valid_bloq_decomposition


def test_prepare_nu():
    num_bits_p = 6
    m_param = 2 ** (2 * num_bits_p + 3)
    prep = PrepareNuState(num_bits_p, m_param)
    assert_valid_bloq_decomposition(prep)
    prep = PrepareNuState(num_bits_p, m_param).adjoint()
    assert_valid_bloq_decomposition(prep)


def test_prepare_nu_t_counts():
    num_bits_p = 6
    m_param = 2 ** (2 * num_bits_p + 3)
    num_bits_m = (m_param - 1).bit_length()
    # arithmetic + inequality + 3 Toffolis for flag (eq 89 + sentence immediately following it)
    expected_cost = 3 * num_bits_p**2 + num_bits_p + 4 * num_bits_m * (num_bits_p + 1) + 4
    # factor of two for inverserse prepare, controlled hadamard + testing on nu.
    expected_cost += 2 * 4 * (num_bits_p - 1) + 6 * num_bits_p + 2
    eq_90 = 3 * num_bits_p**2 + 15 * num_bits_p - 7 + 4 * num_bits_m * (num_bits_p + 1)
    assert expected_cost == eq_90 + 5
    prep = PrepareNuState(num_bits_p, m_param)
    qual_cost = get_cost_value(prep, QECGatesCost()).total_t_count()
    prep = PrepareNuState(num_bits_p, m_param).adjoint()
    qual_cost += get_cost_value(prep, QECGatesCost()).total_t_count()
    qual_cost //= 4
    comp_diff = 1
    assert qual_cost == expected_cost - comp_diff
