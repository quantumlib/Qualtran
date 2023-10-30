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
from qualtran.bloqs.chemistry.pbc.first_quantization.select_uv import SelectUVFirstQuantization


def _make_select_uv():
    from qualtran.bloqs.chemistry.pbc.first_quantization import SelectUVFirstQuantization

    num_bits_p = 5
    eta = 10
    num_bits_nuc_pos = 16

    sel = SelectUVFirstQuantization(
        num_bits_p=num_bits_p, eta=eta, num_bits_nuc_pos=num_bits_nuc_pos
    )
    return sel


def test_select_uv_t_counts():
    num_bits_p = 6
    eta = 10
    num_bits_nuc_pos = 8
    expected_cost = 24 * num_bits_p + 3 * (
        2 * num_bits_p * num_bits_nuc_pos - num_bits_p * (num_bits_p + 1) - 1
    )
    sel = SelectUVFirstQuantization(num_bits_p, eta, num_bits_nuc_pos)
    _, counts = sel.call_graph()
    print(counts)
    qual_cost = counts[TGate()] // 4
    # + 6 as they cost additon as nbits not nbits - 1, there are 6 additions / subtractions.
    assert qual_cost + 6 == expected_cost
