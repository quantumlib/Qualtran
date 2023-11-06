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
from qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_and_prepare import (
    _prep_first_quant,
    _sel_first_quant,
    PrepareFirstQuantizationWithProj,
    SelectFirstQuantizationWithProj,
)
from qualtran.testing import execute_notebook

# def test_notebook():
#     execute_notebook('first_quantization')


def test_prepare(bloq_autotester):
    bloq_autotester(_prep_first_quant)


def test_select(bloq_autotester):
    bloq_autotester(_sel_first_quant)


# def test_select_t_costs():
#     num_bits_p = 6
#     eta = 10
#     num_atoms = 10
#     lambda_zeta = 10
#     num_bits_nuc_pos = 41
#     cost = 0

#     sel_first_quant = SelectFirstQuantization(
#         num_bits_p, eta, num_atoms, lambda_zeta, num_bits_nuc_pos=num_bits_nuc_pos
#     )
#     cost += sel_first_quant.call_graph()[1][TGate()]

#     expected_cost = 7 * (12 * eta * num_bits_p) + 4 * (4 * eta - 8)
#     expected_cost += 4 * (5 * (num_bits_p - 1) + 2)
#     expected_cost += 4 * (24 * num_bits_p)
#     expected_cost += 4 * (
#         3 * (2 * num_bits_p * num_bits_nuc_pos - num_bits_p * (num_bits_p + 1) - 1)
#     )
#     cost += 4 * 6
#     assert cost == expected_cost


def test_prepare_t_costs():
    num_bits_p = 6
    num_bits_n = 8
    eta = 10
    num_atoms = 10
    lambda_zeta = 10
    num_bits_nuc_pos = 16
    b_r = 8
    num_bits_m = 19
    num_bits_t = 16
    cost = 0
    prep_first_quant = PrepareFirstQuantizationWithProj(
        num_bits_p,
        num_bits_n,
        eta,
        num_atoms,
        lambda_zeta,
        m_param=2**num_bits_m,
        num_bits_nuc_pos=num_bits_nuc_pos,
        num_bits_rot_aa=b_r,
        num_bits_t=num_bits_t,
    )
    cost += prep_first_quant.call_graph()[1][TGate()] // 4
    prep_first_quant = PrepareFirstQuantizationWithProj(
        num_bits_p,
        num_bits_n,
        eta,
        num_atoms,
        lambda_zeta,
        num_bits_nuc_pos=num_bits_nuc_pos,
        m_param=2**num_bits_m,
        num_bits_rot_aa=b_r,
        num_bits_t=num_bits_t,
        adjoint=True,
    )
    cost += prep_first_quant.call_graph()[1][TGate()] // 4
    n_eta = (eta - 1).bit_length()
    expected_cost = 6 * num_bits_t + 2  # C1
    expected_cost += 14 * n_eta + 8 * b_r - 36  # C2
    expected_cost += 2 * (2 * num_bits_n + 9) + 2 * (num_bits_n - num_bits_p) + 20  # C3
    expected_cost += 3 * num_bits_n**2 + num_bits_n + 4 * num_bits_m * (num_bits_n + 1) + 4  # C4
    expected_cost += (
        2 * 4 * (num_bits_n - 1) + (num_bits_n - num_bits_p - 1) + 6 * num_bits_n + 2 + 2
    )  # C4
    cost += 1  # comparator off by one
    expected_cost += lambda_zeta + int(np.ceil(lambda_zeta**0.5))
    assert cost == expected_cost
