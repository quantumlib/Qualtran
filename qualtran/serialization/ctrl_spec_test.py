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

from qualtran import CtrlSpec, QBit, QUInt
from qualtran.serialization.ctrl_spec import ctrl_spec_from_proto, ctrl_spec_to_proto


@pytest.mark.parametrize(
    "ctrl_spec",
    [
        CtrlSpec(),
        CtrlSpec(cvs=0),
        CtrlSpec(qdtypes=QUInt(4), cvs=0b0110),
        CtrlSpec(cvs=[0, 1, 1, 0]),
        CtrlSpec(qdtypes=[QBit(), QBit()], cvs=[[0, 1], [1, 0]]),
    ],
)
def test_ctrl_spec_to_proto_roundtrip(ctrl_spec: CtrlSpec):
    ctrl_spec_proto = ctrl_spec_to_proto(ctrl_spec)
    ctrl_spec_roundtrip = ctrl_spec_from_proto(ctrl_spec_proto)
    assert ctrl_spec == ctrl_spec_roundtrip
