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

import attrs
import pytest

import qualtran.testing as qlt_testing
from qualtran import CtrlSpec
from qualtran.bloqs.block_encoding.lcu_block_encoding import (
    _black_box_lcu_block,
    _black_box_select_block,
    _lcu_block,
    _select_block,
)
from qualtran.resource_counting import GateCounts, get_cost_value, QECGatesCost


def test_lcu_block_encoding(bloq_autotester):
    bloq_autotester(_lcu_block)


def test_black_box_lcu_block_encoding(bloq_autotester):
    bloq_autotester(_black_box_lcu_block)


def test_select_block_encoding(bloq_autotester):
    bloq_autotester(_select_block)


def test_black_box_select_block_encoding(bloq_autotester):
    bloq_autotester(_black_box_select_block)


def test_ctrl_lcu_be_cost():
    bloq = _lcu_block()
    assert bloq.controlled() == attrs.evolve(bloq, control_val=1)
    assert bloq.controlled(CtrlSpec(cvs=0)) == attrs.evolve(bloq, control_val=0)

    assert get_cost_value(bloq.controlled(), QECGatesCost()) == GateCounts(
        cswap=28, and_bloq=77, clifford=438, rotation=16, measurement=77
    )


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('lcu_block_encoding')
