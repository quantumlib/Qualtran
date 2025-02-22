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
import pytest

import qualtran.testing as qlt_testing
from qualtran.bloqs.basic_gates import Swap
from qualtran.resource_counting import get_cost_value, QECGatesCost

from .kikuchi_block_encoding import _kikuchi_matrix, _kikuchi_matrix_symb


@pytest.mark.parametrize("bloq_ex", [_kikuchi_matrix, _kikuchi_matrix_symb])
def test_examples(bloq_autotester, bloq_ex):
    if bloq_autotester.check_name == 'serialize':
        pytest.skip()

    bloq_autotester(bloq_ex)


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('kikuchi_block_encoding')


def test_controlled_cost():
    bloq = _kikuchi_matrix()
    _, sigma = bloq.call_graph(max_depth=2)
    _, ctrl_sigma = bloq.controlled().call_graph(max_depth=2)

    assert set(sigma.items()) - set(ctrl_sigma.items()) == {(Swap(32), 1), (Swap(1), 1)}
    assert set(ctrl_sigma.items()) - set(sigma.items()) == {
        (Swap(32).controlled(), 1),
        (Swap(1).controlled(), 1),
    }


def test_cost():
    bloq = _kikuchi_matrix()

    _ = get_cost_value(bloq, QECGatesCost())


def test_cost_symb():
    bloq = _kikuchi_matrix_symb()

    _ = get_cost_value(bloq, QECGatesCost())
    print(_)
