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
from qualtran.bloqs.basic_gates import XGate
from qualtran.bloqs.for_testing.matrix_gate import MatrixGate
from qualtran.bloqs.mcmt.controlled_via_and import (
    _controlled_via_and_ints,
    _controlled_via_and_qbits,
    ControlledViaAnd,
)
from qualtran.resource_counting import GateCounts, get_cost_value, QECGatesCost


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

    ctrl_bloq = ControlledViaAnd(subbloq, ctrl_spec)
    naive_ctrl_bloq = Controlled(subbloq, ctrl_spec)

    expected_tensor = naive_ctrl_bloq.tensor_contract()
    actual_tensor = ctrl_bloq.tensor_contract()

    np.testing.assert_allclose(expected_tensor, actual_tensor)


def test_nested_controls():
    spec1 = CtrlSpec(QUInt(4), [2, 3])
    spec2 = CtrlSpec(QInt(4), [1, 2])
    spec = CtrlSpec((QInt(4), QUInt(4)), ([1, 2], [2, 3]))

    rs = np.random.RandomState(42)
    bloq = MatrixGate.random(2, random_state=rs)

    ctrl_bloq = ControlledViaAnd(bloq, spec1).controlled(ctrl_spec=spec2)
    assert ctrl_bloq == ControlledViaAnd(bloq, spec)


def test_nested_controlled_x():
    bloq = XGate()

    ctrl_bloq = ControlledViaAnd(bloq, CtrlSpec(cvs=[1, 1])).controlled(
        ctrl_spec=CtrlSpec(cvs=[1, 1])
    )
    cost = get_cost_value(ctrl_bloq, QECGatesCost())

    n_ands = 3
    assert cost == GateCounts(and_bloq=n_ands, clifford=n_ands + 1, measurement=n_ands)

    np.testing.assert_allclose(
        ctrl_bloq.tensor_contract(),
        XGate().controlled(CtrlSpec(cvs=[1, 1, 1, 1])).tensor_contract(),
    )
