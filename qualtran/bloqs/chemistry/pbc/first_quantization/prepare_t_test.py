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
from qualtran.bloqs.chemistry.pbc.first_quantization.prepare import (
    UniformSuperpostionIJFirstQuantization,
)
from qualtran.bloqs.chemistry.pbc.first_quantization.prepare_t import PrepareTFirstQuantization
from qualtran.testing import assert_valid_bloq_decomposition


def _make_prepare_t():
    from qualtran.bloqs.chemistry.pbc.first_quantization import PrepareTFirstQuantization

    num_bits_p = 5
    eta = 10

    return PrepareTFirstQuantization(num_bits_p=num_bits_p, eta=eta)


def test_prepare_t():
    num_bits_p = 6
    eta = 10
    prep = PrepareTFirstQuantization(num_bits_p, eta)
    assert_valid_bloq_decomposition(prep)


def _make_prepare_t():
    from qualtran.bloqs.chemistry.pbc.first_quantization import PrepareTFirstQuantization

    num_bits_p = 5
    eta = 10

    return PrepareTFirstQuantization(num_bits_p=num_bits_p, eta=eta)


def test_prepare_kinetic_t_counts():
    num_bits_p = 6
    eta = 10
    b_r = 8
    n_eta = (eta - 1).bit_length()
    expected_cost = (14 * n_eta + 8 * b_r - 36) + 2 * (2 * num_bits_p + 9)
    uni = UniformSuperpostionIJFirstQuantization(eta, num_bits_rot_aa=b_r)
    _, counts = uni.call_graph()
    qual_cost = counts[TGate()]
    uni = UniformSuperpostionIJFirstQuantization(eta, num_bits_rot_aa=b_r, adjoint=True)
    _, counts = uni.call_graph()
    qual_cost += counts[TGate()]
    prep = PrepareTFirstQuantization(num_bits_p, eta, num_bits_rot_aa=b_r)
    _, counts = prep.call_graph()
    qual_cost += counts[TGate()]
    prep = PrepareTFirstQuantization(num_bits_p, eta, num_bits_rot_aa=b_r, adjoint=True)
    _, counts = prep.call_graph()
    qual_cost += counts[TGate()]
    qual_cost //= 4
    assert qual_cost == expected_cost
