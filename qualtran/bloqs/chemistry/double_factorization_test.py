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
from qualtran.bloqs.chemistry.double_factorization import DoubleFactorization
from qualtran.resource_counting import get_bloq_counts_graph
from qualtran.testing import assert_valid_bloq_decomposition, execute_notebook


def _make_double_factorization():
    from qualtran.bloqs.chemistry.double_factorization import DoubleFactorization

    return DoubleFactorization(10, 20, 8)


def test_double_factorization():
    df = DoubleFactorization(10, 12, 8)
    assert_valid_bloq_decomposition(df)


def test_double_factorization_counts_graph():
    graph, sigma = get_bloq_counts_graph(DoubleFactorization(4, 10, 4))
    assert sigma[TGate()] == 4656


def test_notebook():
    execute_notebook("double_factorization")
