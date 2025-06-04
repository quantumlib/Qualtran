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
from qualtran.drawing.musical_score import pretty_format_soq_text

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


def test_pretty_format_soq_text():
    soq_text = "Z^2*2**(11 - Min(6, ceiling(log2(6283185307.17959*Abs(Y)))))*Y"
    expected_str = "Z^64*Y"
    assert expected_str == pretty_format_soq_text(soq_text)
    soq_text = "Z^2*Y/2**Min(6, ceiling(log2(6283185307.17959*Abs(Y))))"
    expected_str = "Z^Y/32"
    assert expected_str == pretty_format_soq_text(soq_text)


@pytest.mark.notebook
def test_notebook():
    execute_notebook('musical_score')
