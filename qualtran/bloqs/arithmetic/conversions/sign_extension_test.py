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

from qualtran import BloqBuilder, QInt, QUInt
from qualtran.bloqs.arithmetic.conversions.sign_extension import (
    _sign_extend,
    _sign_extend_fxp,
    SignExtend,
)
from qualtran.bloqs.basic_gates import IntEffect, IntState


def test_examples(bloq_autotester):
    bloq_autotester(_sign_extend)
    bloq_autotester(_sign_extend_fxp)


@pytest.mark.parametrize("l, r", [(2, 4)])
def test_sign_extend_tensor(l: int, r: int):
    bloq = SignExtend(QInt(l), QInt(r))

    def _as_unsigned(num: int, bitsize: int):
        # TODO remove this once IntState supports signed values
        return QUInt(bitsize).from_bits(QInt(bitsize).to_bits(num))

    for x in QInt(l).get_classical_domain():
        bb = BloqBuilder()
        qx = bb.add(IntState(_as_unsigned(x, l), l))
        qx = bb.add(bloq, x=qx)
        bb.add(IntEffect(_as_unsigned(x, r), r), val=qx)
        cbloq = bb.finalize()

        np.testing.assert_allclose(cbloq.tensor_contract(), 1)
