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
from qualtran.bloqs.block_encoding.chebyshev_polynomial import (
    _black_box_chebyshev_poly,
    _chebyshev_poly,
)


def test_chebyshev(bloq_autotester):
    bloq_autotester(_chebyshev_poly)


def test_black_box_chebyshev(bloq_autotester):
    bloq_autotester(_black_box_chebyshev_poly)


def test_chebyshev_t_counts():
    counts = _chebyshev_poly().call_graph()[1]
    counts_decomp = _chebyshev_poly().decompose_bloq().call_graph()[1]
    assert counts[TGate()] == counts_decomp[TGate()]
