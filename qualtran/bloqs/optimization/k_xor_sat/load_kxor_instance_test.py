#  Copyright 2024 Google LLC
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
from unittest.mock import ANY

import pytest

import qualtran.testing as qlt_testing
from qualtran import Bloq
from qualtran.bloqs.optimization.k_xor_sat.load_kxor_instance import _load_scopes, _load_scopes_symb
from qualtran.resource_counting import GateCounts, get_cost_value, QECGatesCost


@pytest.mark.parametrize("bloq", [_load_scopes, _load_scopes_symb], ids=lambda be: be.name)
def test_examples(bloq_autotester, bloq: Bloq):
    bloq_autotester(bloq)


def test_load_instance():
    bloq = _load_scopes()

    gc = get_cost_value(bloq, QECGatesCost())
    assert gc == GateCounts(and_bloq=3, clifford=ANY, measurement=ANY)

    # classical action
    for j, (S, _) in enumerate(tuple(bloq.inst.batched_scopes)):  # type: ignore
        assert bloq.call_classically(j=j) == (j, bloq.inst.scope_as_int(S))


def test_load_instance_cost_symb():
    bloq = _load_scopes_symb()

    m, k = bloq.inst.m, bloq.inst.k
    logn = bloq.inst.index_bitsize
    gc = get_cost_value(bloq, QECGatesCost())
    assert gc == GateCounts(and_bloq=m - 2, clifford=k * m * logn + m - 2, measurement=m - 2)


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('load_kxor_instance')
