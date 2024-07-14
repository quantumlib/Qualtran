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

from qualtran import CtrlSpec
from qualtran.bloqs.mcmt.ctrl_spec_activation import (
    _ctrl_on_bits,
    _ctrl_on_int,
    _ctrl_on_multiple_values,
    _ctrl_on_nd_bits,
    CtrlSpecActivation,
)


@pytest.mark.parametrize(
    "example", [_ctrl_on_bits, _ctrl_on_nd_bits, _ctrl_on_int, _ctrl_on_multiple_values]
)
def test_examples(bloq_autotester, example):
    bloq_autotester(example)


@pytest.mark.parametrize("ctrl_spec", [CtrlSpec(), CtrlSpec(cvs=0), CtrlSpec(cvs=[0])])
def test_raises_for_single_qubit_controls(ctrl_spec: CtrlSpec):
    with pytest.raises(ValueError):
        _ = CtrlSpecActivation(ctrl_spec)
