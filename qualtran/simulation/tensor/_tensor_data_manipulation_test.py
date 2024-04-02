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

from qualtran import CtrlSpec, QAny, Register, Side, Signature
from qualtran.simulation.tensor import (
    active_space_for_ctrl_spec,
    eye_tensor_for_signature,
    tensor_out_inp_shape_from_signature,
    tensor_shape_from_signature,
)


def test_tensor_shape_from_signature():
    # Test trivial case
    assert tensor_shape_from_signature(Signature.build(x=1)) == (2, 2)

    # Test left / right / thru cases
    left_register = Register('left', QAny(1), side=Side.LEFT, shape=())
    right_register = Register('right', QAny(2), side=Side.RIGHT, shape=(2,))
    thru_register = Register('thru', QAny(3), side=Side.THRU, shape=(2, 2))

    assert tensor_shape_from_signature(Signature([left_register])) == (2,)
    assert tensor_shape_from_signature(Signature([right_register])) == (4, 4)
    assert tensor_shape_from_signature(Signature([thru_register])) == (8, 8, 8, 8) * 2

    # Test all 3 and asser that ordering is preserved
    signature = Signature([left_register, right_register, thru_register])
    inp_shape = (2,) + (8, 8, 8, 8)
    out_shape = (4, 4) + (8, 8, 8, 8)
    assert tensor_shape_from_signature(signature) == out_shape + inp_shape
    assert tensor_out_inp_shape_from_signature(signature) == (out_shape, inp_shape)


def test_eye_tensor_for_signature():
    # Test left / right / thru cases individually
    left_register = Register('left', QAny(1), side=Side.LEFT)
    right_register = Register('right', QAny(1), side=Side.RIGHT)
    thru_register = Register('thru', QAny(1), side=Side.THRU)
    left_data = right_data = np.array([1, 0])
    np.testing.assert_allclose(eye_tensor_for_signature(Signature([left_register])), left_data)
    np.testing.assert_allclose(eye_tensor_for_signature(Signature([right_register])), right_data)
    thru_data = np.eye(2)
    np.testing.assert_allclose(eye_tensor_for_signature(Signature([thru_register])), thru_data)

    # Test LEFT + RIGHT case
    actual = eye_tensor_for_signature(Signature([left_register, right_register]))
    expected = np.array([[1, 0], [0, 0]])  # 1 only when LEFT is 0 and RIGHT is 0
    np.testing.assert_allclose(actual, expected)

    # It's helpful to take einsum for more complicated cases, as shown below, because each
    # variable corresponds to an output / input index.

    # Test LEFT + THRU case
    actual = eye_tensor_for_signature(Signature([left_register, thru_register]))
    expected = np.einsum('jk,i->jik', thru_data, left_data)
    np.testing.assert_allclose(actual, expected)

    # Test RIGHT + THRU case
    actual = eye_tensor_for_signature(Signature([right_register, thru_register]))
    expected = np.einsum('jk,i->ijk', thru_data, right_data)
    np.testing.assert_allclose(actual, expected)

    # Test LEFT + RIGHT + THRU case
    actual = eye_tensor_for_signature(Signature([left_register, right_register, thru_register]))
    expected = np.einsum('i,jk,l->ijlk', right_data, thru_data, left_data)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize('cv', [0, 1])
def test_active_space_for_ctrl_spec(cv: int):
    ctrl_spec = CtrlSpec(cvs=cv)
    signature = Signature([Register('ctrl', QAny(1)), Register('q', QAny(1))])
    assert active_space_for_ctrl_spec(signature, ctrl_spec) == (cv, slice(2), cv, slice(2))
    signature = Signature([Register('ctrl', QAny(1)), Register('q', QAny(1), side=Side.LEFT)])
    assert active_space_for_ctrl_spec(signature, ctrl_spec) == (cv, cv, slice(2))
    signature = Signature([Register('ctrl', QAny(1)), Register('q', QAny(1), side=Side.RIGHT)])
    assert active_space_for_ctrl_spec(signature, ctrl_spec) == (cv, slice(2), cv)
