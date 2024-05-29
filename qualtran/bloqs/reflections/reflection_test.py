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
import pytest

from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.basic_gates.t_gate import TGate
from qualtran.bloqs.reflections.reflection import _reflection
from qualtran.testing import execute_notebook


def test_reflection(bloq_autotester):
    bloq_autotester(_reflection)


def test_reflection_t_counts():
    counts = _reflection().call_graph()[1]
    counts_decomp = _reflection().decompose_bloq().call_graph()[1]
    assert counts[TGate()] == counts_decomp[TGate()]


@pytest.mark.notebook
def test_notebook():
    execute_notebook('reflections')
