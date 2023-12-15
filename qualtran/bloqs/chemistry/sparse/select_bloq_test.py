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
from qualtran.bloqs.chemistry.sparse.select_bloq import _sel_sparse


def test_prep_inner(bloq_autotester):
    bloq_autotester(_sel_sparse)


def test_decompose_bloq_counts():
    sel = _sel_sparse()
    cost_decomp = sel.decompose_bloq().call_graph()[1][TGate()]
    cost_call = sel.call_graph()[1][TGate()]
    assert cost_call == cost_decomp
