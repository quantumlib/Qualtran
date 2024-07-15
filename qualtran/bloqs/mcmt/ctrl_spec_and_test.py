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
import numpy as np
import pytest

from qualtran import CtrlSpec, QUInt
from qualtran.bloqs.mcmt.ctrl_spec_and import (
    _ctrl_on_bits,
    _ctrl_on_int,
    _ctrl_on_multiple_values,
    _ctrl_on_nd_bits,
    CtrlSpecAnd,
)
from qualtran.simulation.classical_sim import get_classical_truth_table


@pytest.mark.parametrize(
    "example", [_ctrl_on_bits, _ctrl_on_nd_bits, _ctrl_on_int, _ctrl_on_multiple_values]
)
def test_examples(bloq_autotester, example):
    bloq_autotester(example)


@pytest.mark.parametrize("ctrl_spec", [CtrlSpec(), CtrlSpec(cvs=0), CtrlSpec(cvs=[0])])
def test_raises_for_single_qubit_controls(ctrl_spec: CtrlSpec):
    with pytest.raises(ValueError):
        _ = CtrlSpecAnd(ctrl_spec)


@pytest.mark.parametrize(
    "ctrl_spec",
    [CtrlSpec(QUInt(4), cvs=4), CtrlSpec((QUInt(4), QUInt(4)), cvs=(np.array(3), np.array(5)))],
)
def test_truth_table_using_classical_sim(ctrl_spec: CtrlSpec):
    bloq = CtrlSpecAnd(ctrl_spec)
    _, _, tt = get_classical_truth_table(bloq)
    for in_vals, out_vals in tt:
        # check: control values not modified
        assert all(ctrl_in == ctrl_out for ctrl_in, ctrl_out in zip(in_vals, out_vals))

        # check: target bit (last output value) matches `is_active`
        assert out_vals[-1] == ctrl_spec.is_active(*in_vals)
