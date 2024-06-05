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
    prepare = StatePreparationAliasSampling.from_lcu_probs(coeff, probability_epsilon=0.1)
    bloq = ReflectionUsingPrepare(prepare, global_phase=-1j)
    _assert_bloq_has_qpic_diagram(
        bloq,
        r"""
DEFINE off color=white
DEFINE on color=black
selection W \textrm{\scalebox{0.8}{selection}}
selection / \textrm{\scalebox{0.5}{BoundedQUInt(2, 4)}}
LABEL length=10
selection G:width=17:shape=box \textrm{\scalebox{0.8}{R\_L}}
""",
    )

    _assert_bloq_has_qpic_diagram(
        bloq.decompose_bloq(),
        r"""
DEFINE off color=white
DEFINE on color=black
_empty_wire W off
selection W \textrm{\scalebox{0.8}{selection}}
selection / \textrm{\scalebox{0.5}{BoundedQUInt(2, 4)}}
LABEL length=10
reg W off
reg_1 W off
reg_2 W off
reg_3 W off
reg_4 W off
reg[0] W off
reg[1] W off
reg_5 W off
reg:on G:width=25:shape=box \textrm{\scalebox{0.8}{alloc}}
reg / \textrm{\scalebox{0.5}{QAny(2)}}
reg_1:on G:width=25:shape=box \textrm{\scalebox{0.8}{alloc}}
reg_1 / \textrm{\scalebox{0.5}{QAny(2)}}
reg_2:on G:width=25:shape=box \textrm{\scalebox{0.8}{alloc}}
reg_2 / \textrm{\scalebox{0.5}{QAny(2)}}
reg_3:on G:width=25:shape=box \textrm{\scalebox{0.8}{alloc}}
reg_3 / \textrm{\scalebox{0.5}{QAny(1)}}
reg_4:on G:width=25:shape=box \textrm{\scalebox{0.8}{alloc}}
reg_4 / \textrm{\scalebox{0.5}{QAny(1)}}
_empty_wire G:width=65:shape=8 GPhase((-0-1j))
selection G:width=121:shape=box \textrm{\scalebox{0.8}{StatePreparationAliasSampling}} reg G:width=37:shape=box \textrm{\scalebox{0.8}{sigma\_mu}} reg_1 G:width=17:shape=box \textrm{\scalebox{0.8}{alt}} reg_2 G:width=21:shape=box \textrm{\scalebox{0.8}{keep}} reg_3 G:width=65:shape=box \textrm{\scalebox{0.8}{less\_than\_equal}}
+reg_4
selection:off G:width=5:shape=> \textrm{\scalebox{0.8}{}} reg[0]:on G:width=17:shape=box \textrm{\scalebox{0.8}{[0]}} reg[1]:on G:width=17:shape=box \textrm{\scalebox{0.8}{[1]}}
reg_4 G:width=9:shape=box \textrm{\scalebox{0.8}{Z}} -reg[0] -reg[1]
+reg_4
reg[0]:off G:width=17:shape=box \textrm{\scalebox{0.8}{[0]}} reg[1]:off G:width=17:shape=box \textrm{\scalebox{0.8}{[1]}} reg_5:on G:width=5:shape=< \textrm{\scalebox{0.8}{}}
reg_5 / \textrm{\scalebox{0.5}{BoundedQUInt(2, 4)}}
reg_4:off G:width=21:shape=box \textrm{\scalebox{0.8}{free}}
reg_5 G:width=121:shape=box \textrm{\scalebox{0.8}{StatePreparationAliasSampling}} reg G:width=37:shape=box \textrm{\scalebox{0.8}{sigma\_mu}} reg_1 G:width=17:shape=box \textrm{\scalebox{0.8}{alt}} reg_2 G:width=21:shape=box \textrm{\scalebox{0.8}{keep}} reg_3 G:width=65:shape=box \textrm{\scalebox{0.8}{less\_than\_equal}}
reg:off G:width=21:shape=box \textrm{\scalebox{0.8}{free}}
reg_1:off G:width=21:shape=box \textrm{\scalebox{0.8}{free}}
reg_2:off G:width=21:shape=box \textrm{\scalebox{0.8}{free}}
reg_3:off G:width=21:shape=box \textrm{\scalebox{0.8}{free}}""",
    )
