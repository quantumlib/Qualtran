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

from qualtran import Bloq
from qualtran.bloqs.reflections.reflection_using_prepare import ReflectionUsingPrepare
from qualtran.bloqs.state_preparation import StatePreparationAliasSampling
from qualtran.drawing.qpic_diagram import get_qpic_data


def _assert_bloq_has_qpic_diagram(bloq: Bloq, expected_qpic_data: str):
    qpic_data = get_qpic_data(bloq)
    qpic_data_str = '\n'.join(qpic_data)
    assert qpic_data_str.strip() == expected_qpic_data.strip()


def test_qpic_data_for_reflect_using_prepare():
    coeff = [0.1, 0.2, 0.3, 0.4]
    prepare = StatePreparationAliasSampling.from_probabilities(coeff, precision=0.1)
    bloq = ReflectionUsingPrepare(prepare, global_phase=-1j)
    _assert_bloq_has_qpic_diagram(
        bloq,
        r"""
DEFINE off color=white
DEFINE on color=black
selection W \textrm{\scalebox{0.8}{selection}}
selection / \textrm{\scalebox{0.5}{BQUInt(2, 4)}}
LABEL length=10
selection G:width=17:shape=box \textrm{\scalebox{0.8}{R\_L}}
""",
    )
