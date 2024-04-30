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
from qualtran.bloqs.state_preparation import PrepareUniformSuperposition
from qualtran.drawing.qpic_diagram import get_qpic_data


def _assert_bloq_has_qpic_diagram(bloq: Bloq, expected_qpic_data: str):
    qpic_data = get_qpic_data(bloq)
    qpic_data_str = '\n'.join(qpic_data)
    assert qpic_data_str.strip() == expected_qpic_data.strip()


def test_qpic_data_for_uniform_superposition():
    _assert_bloq_has_qpic_diagram(
        PrepareUniformSuperposition(3),
        r"""
DEFINE off color=white
DEFINE on color=black
target W \textrm{target}
target / \textrm{\scalebox{0.5}{QAny(2)}}
LABEL length=10
target G:width=55 \textrm{UNIFORM(3)}
""",
    )

    _assert_bloq_has_qpic_diagram(
        PrepareUniformSuperposition(3).decompose_bloq(),
        r"""
DEFINE off color=white
DEFINE on color=black
target W \textrm{target}
target / \textrm{\scalebox{0.5}{QAny(2)}}
LABEL length=10
reg W off
reg[0] W off
reg[1] W off
reg_1 W off
reg:on G:width=30 \textrm{alloc}
target:off G:width=10 \textrm{S} reg[0]:on G:width=20 \textrm{[0]} reg[1]:on G:width=20 \textrm{[1]}
reg[0] G:width=10 \textrm{H}
reg[1] G:width=10 \textrm{H}
reg[0]:off G:width=20 \textrm{[0]} reg[1]:off G:width=20 \textrm{[1]} reg_1:on G:width=10 \textrm{J}
reg_1 G:width=30 \textrm{In(x)} reg G:width=45 \textrm{$\oplus$(x $<$ 3)}
reg G:width=55 \textrm{Rz(0.392π)}
reg_1 G:width=30 \textrm{In(x)} reg G:width=45 \textrm{$\oplus$(x $<$ 3)}
reg_1:off G:width=10 \textrm{S} reg[0]:on G:width=20 \textrm{[0]} reg[1]:on G:width=20 \textrm{[1]}
reg:off G:width=25 \textrm{free}
reg[0] G:width=10 \textrm{H}
reg[1] G:width=10 \textrm{H}
target:on G:width=10 \textrm{\&} -reg[0] -reg[1]
target G:width=55 \textrm{Rz(0.392π)}
target:off G:width=10 \textrm{\&} -reg[0] -reg[1]
reg[0] G:width=10 \textrm{H}
reg[1] G:width=10 \textrm{H}
reg[0]:off G:width=20 \textrm{[0]} reg[1]:off G:width=20 \textrm{[1]} reg:on G:width=10 \textrm{J}""".strip(),
    )
