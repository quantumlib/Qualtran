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

from qualtran import QInt, QUInt
from qualtran.bloqs.arithmetic.negate import _negate, _negate_symb, Negate


def test_examples(bloq_autotester):
    bloq_autotester(_negate)
    bloq_autotester(_negate_symb)


@pytest.mark.parametrize("bitsize", [1, 2, 3, 4, 8])
def test_negate_classical_sim(bitsize: int):
    # TODO use QInt once classical sim is fixed.
    #      classical sim currently only supports unsigned.
    def _uint_to_int(val: int) -> int:
        return QInt(bitsize).from_bits(QUInt(bitsize).to_bits(val))

    dtype = QUInt(bitsize)

    bloq = Negate(dtype)
    for x_unsigned in dtype.get_classical_domain():
        (neg_x_unsigned,) = bloq.call_classically(x=x_unsigned)
        x = _uint_to_int(x_unsigned)
        neg_x = _uint_to_int(int(neg_x_unsigned))
        if x == -(2 ** (bitsize - 1)):
            # twos complement negate(-2**(n - 1)) == -2**(n - 1)
            assert neg_x == x
        else:
            # all other values
            assert neg_x == -x
