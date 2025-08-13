#  Copyright 2025 Google LLC
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

from qualtran import CBit, CtrlSpec
from qualtran.bloqs.basic_gates import XGate
from qualtran.bloqs.mcmt.classically_controlled import ClassicallyControlled


def test_controlled_x_simulated() -> None:
    bloq = XGate()

    ctrl_bloq = ClassicallyControlled(bloq, CtrlSpec(CBit(), cvs=1))
    assert len(ctrl_bloq.ctrl_reg_names) == 1
    inputs = {ctrl_bloq.ctrl_reg_names[0]: 0, 'q': 0}
    assert ctrl_bloq.call_classically(**inputs) == (0, 0)
    inputs = {ctrl_bloq.ctrl_reg_names[0]: 0, 'q': 1}
    assert ctrl_bloq.call_classically(**inputs) == (0, 1)
    inputs = {ctrl_bloq.ctrl_reg_names[0]: 1, 'q': 0}
    assert ctrl_bloq.call_classically(**inputs) == (1, 1)
    inputs = {ctrl_bloq.ctrl_reg_names[0]: 1, 'q': 1}
    assert ctrl_bloq.call_classically(**inputs) == (1, 0)
