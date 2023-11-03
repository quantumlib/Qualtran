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

from qualtran.bloqs.chemistry.pbc.first_quantization.first_quantization import (
    _prep_first_quant,
    _sel_first_quant,
    SelectFirstQuantization,
)
from qualtran.testing import assert_valid_bloq_decomposition, execute_notebook

# def test_notebook():
#     execute_notebook('first_quantization')


def test_prepare(bloq_autotester):
    bloq_autotester(_prep_first_quant)


def test_select(bloq_autotester):
    # bloq_autotester(_sel_first_quant)
    num_bits_p = 6
    eta = 10
    num_atoms = 10
    lambda_zeta = 10
    sel_first_quant = SelectFirstQuantization(num_bits_p, eta, num_atoms, lambda_zeta)

    assert_valid_bloq_decomposition(sel_first_quant)
