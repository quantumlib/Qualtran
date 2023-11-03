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

from qualtran.bloqs.basic_gates import TGate, Toffoli
from qualtran.bloqs.chemistry.pbc.first_quantization.first_quantization import (
    _prep_first_quant,
    _sel_first_quant,
    PrepareFirstQuantization,
    SelectFirstQuantization,
)
from qualtran.testing import assert_valid_bloq_decomposition, execute_notebook

# def test_notebook():
#     execute_notebook('first_quantization')


def test_prepare(bloq_autotester):
    bloq_autotester(_prep_first_quant)


def test_select(bloq_autotester):
    bloq_autotester(_sel_first_quant)


def test_select_costs():
    num_bits_p = 6
    eta = 10
    num_atoms = 10
    lambda_zeta = 10
    num_bits_nuc_pos = 41
    m_param = 2**8
    cost = 0

    def keeper(bloq):
        if isinstance(bloq, Toffoli):
            return True
        if isinstance(bloq, TGate):
            return True
        return False

    sel_first_quant = SelectFirstQuantization(num_bits_p, eta, num_atoms, lambda_zeta)
    cost += sel_first_quant.call_graph(keep=keeper)[1][Toffoli()]
    cost += sel_first_quant.call_graph(keep=keeper)[1][TGate()] // 7
    from .ft_pw_resource_estimates import pw_qubitization_costs

    print(
        pw_qubitization_costs(
            num_bits_p, eta, 10, 1e-3, (m_param - 1).bit_length(), num_bits_nuc_pos, num_atoms
        )
    )
    print(cost)
