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
from qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_t import (
    _sel_t_proj,
    SelectTFirstQuantizationWithProj,
)


def test_sel_t_proj(bloq_autotester):
    bloq_autotester(_sel_t_proj)


def test_select_kinetic_t_counts():
    num_bits_n = 6
    sel = SelectTFirstQuantizationWithProj(num_bits_n, 10)
    _, counts = sel.call_graph()
    assert counts[TGate()] // 4 == 5 * (num_bits_n - 1) + 2 + 1
