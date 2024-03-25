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
from qualtran import QAny, Register, Side, Signature
from qualtran.simulation.tensor import (
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
