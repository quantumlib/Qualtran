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

from qualtran import BloqBuilder, QUInt
from qualtran.bloqs.arithmetic.bitwise import _cxork, _xor, _xor_symb, _xork, XorK
from qualtran.bloqs.basic_gates import IntEffect, IntState


def test_examples(bloq_autotester):
    bloq_autotester(_cxork)
    bloq_autotester(_xork)


def test_xork_classical_sim():
    k = 0b01101010
    bloq = XorK(QUInt(9), k)
    cbloq = bloq.controlled()

    for x in bloq.dtype.get_classical_domain():
        (x_out,) = bloq.call_classically(x=x)
        assert x_out == x ^ k

        ctrl_out, x_out = cbloq.call_classically(ctrl=0, x=x)
        assert ctrl_out == 0
        assert x_out == x

        ctrl_out, x_out = cbloq.call_classically(ctrl=1, x=x)
        assert ctrl_out == 1
        assert x_out == x ^ k


def test_xor(bloq_autotester):
    bloq_autotester(_xor)


def test_xor_symb(bloq_autotester):
    bloq_autotester(_xor_symb)


def test_xor_call():
    bloq = _xor()
    x, y = bloq.call_classically(x=7, y=0)
    assert x == 7 and y == 7
    x, y = bloq.decompose_bloq().call_classically(x=7, y=0)
    assert x == 7 and y == 7

    bb = BloqBuilder()
    x = bb.add(IntState(13, 4))
    y = bb.add(IntState(0, 4))
    x, y = bb.add_t(bloq, x=x, y=y)
    bb.add(IntEffect(13, 4), val=x)
    bloq = bb.finalize(y=y)

    np.testing.assert_allclose(bloq.tensor_contract(), IntState(13, 4).tensor_contract())
