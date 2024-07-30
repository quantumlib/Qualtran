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

from qualtran import Controlled, CtrlSpec, QInt, QUInt
from qualtran.bloqs.for_testing.matrix_gate import MatrixGate
from qualtran.bloqs.mcmt.controlled_via_and import (
    _controlled_via_and_ints,
    _controlled_via_and_qbits,
    ControlledViaAnd,
)


def test_examples(bloq_autotester):
    bloq_autotester(_controlled_via_and_qbits)
    bloq_autotester(_controlled_via_and_ints)


@pytest.mark.parametrize(
    "ctrl_spec",
    [
        CtrlSpec(QInt(4), [1, -2]),
        CtrlSpec(QUInt(4), [2, 5]),
        CtrlSpec((QInt(2), QUInt(2)), cvs=([1, -2], [2, 3])),
    ],
)
def test_tensor_against_naive_controlled(ctrl_spec: CtrlSpec):
    rs = np.random.RandomState(42)
    subbloq = MatrixGate.random(2, random_state=rs)

    cbloq = ControlledViaAnd(subbloq, ctrl_spec)
    naive_cbloq = Controlled(subbloq, ctrl_spec)

    expected_tensor = naive_cbloq.tensor_contract()
    actual_tensor = cbloq.tensor_contract()

    np.testing.assert_allclose(expected_tensor, actual_tensor)
