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

from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.chemistry.pbc.first_quantization.prepare_uv import (
    _prepare_uv,
    PrepareUVFirstQuantization,
)


def test_prepare_uv(bloq_autotester):
    bloq_autotester(_prepare_uv)


def test_prepare_uv_t_counts():
    num_bits_p = 6
    eta = 10
    num_atoms = 10
    lambda_zeta = 10
    num_bits_nuc_pos = 8
    m_param = 2 ** (2 * num_bits_p + 3)
    num_bits_m = (m_param - 1).bit_length()
    # arithmetic + inequality + 3 Toffolis for flag (eq 89 + sentence immediately following it)
    expected_cost = 3 * num_bits_p**2 + num_bits_p + 4 * num_bits_m * (num_bits_p + 1) + 4
    # factor of two for inverserse prepare, controlled hadamard + testing on nu.
    expected_cost += 2 * 4 * (num_bits_p - 1) + 6 * num_bits_p + 2
    expected_cost += lambda_zeta + int(np.ceil(lambda_zeta**0.5))
    prep = PrepareUVFirstQuantization(
        num_bits_p, eta, num_atoms, m_param, lambda_zeta, num_bits_nuc_pos
    )
    _, counts = prep.call_graph()
    qual_cost = counts[TGate()]
    prep = PrepareUVFirstQuantization(
        num_bits_p, eta, num_atoms, m_param, lambda_zeta, num_bits_nuc_pos, adjoint=True
    )
    _, counts = prep.call_graph()
    qual_cost += counts[TGate()]
    qual_cost //= 4
    comp_diff = 1
    assert qual_cost == expected_cost - comp_diff
