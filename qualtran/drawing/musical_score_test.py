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

from qualtran.bloqs.mcmt import MultiAnd
from qualtran.drawing import dump_musical_score, get_musical_score_data, HLine
from qualtran.testing import execute_notebook


def test_dump_json(tmp_path):
    hline = HLine(y=10, seq_x_start=5, seq_x_end=6)
    assert hline.json_dict() == {
        'y': 10,
        'seq_x_start': 5,
        'seq_x_end': 6,
        'flavor': 'HLineFlavor.QUANTUM',
    }

    cbloq = MultiAnd((1, 1, 0, 1)).decompose_bloq()
    msd = get_musical_score_data(cbloq)
    dump_musical_score(msd, name=f'{tmp_path}/musical_score_example')


def test_musical_score_aligns_with_qubit_count():
    from qualtran.bloqs.for_testing.qubit_count_many_alloc import (
        TestManyAllocAbstracted,
        TestManyAllocOnce,
        TestManyAllocMany,
    )
    from qualtran.resource_counting import get_cost_cache, get_cost_value, QubitCount

    n = 10
    for bloq in [TestManyAllocMany(n), TestManyAllocOnce(n), TestManyAllocAbstracted(n)]:
        expected_qubits = get_cost_value(bloq, QubitCount())
        msd = get_musical_score_data(bloq.decompose_bloq())
        # Ensure qubits (horizontal rows) match qubits from cost values in expected count. This test may depend on how
        # elements of a circuit are represented, and may need to be updated accordingly.
        actual_qubits = msd.max_y + 1
        assert (
            msd.max_y + 1 == expected_qubits
        ), f'{type(bloq).__name__} has too many lines - expected {expected_qubits}; got {msd.max_y + 1}'


@pytest.mark.notebook
def test_notebook():
    execute_notebook('musical_score')
