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


@pytest.mark.parametrize("bitsize", [1, 3, 8])
def test_negate_classical_sim(bitsize: int):
    dtype = QInt(bitsize)
    bloq = Negate(dtype)
    cbloq = bloq.decompose_bloq()

    if bitsize == 3:
        assert list(dtype.get_classical_domain()) == [-4, -3, -2, -1, 0, 1, 2, 3]

    for x_in in dtype.get_classical_domain():
        (x_out,) = cbloq.call_classically(x=x_in)

        if x_in == -(2 ** (bitsize - 1)):
            # twos complement negate(-2**(n - 1)) == -2**(n - 1)
            assert x_out == x_in
        else:
            assert x_out == -x_in


@pytest.mark.parametrize("bitsize", [1, 3, 8])
def test_negate_unsigned_classical_sim(bitsize: int):
    dtype = QUInt(bitsize)
    bloq = Negate(dtype)
    cbloq = bloq.decompose_bloq()

    for x_in in dtype.get_classical_domain():
        (x_out,) = cbloq.call_classically(x=x_in)

        if x_in == 0:
            assert x_out == 0
        else:
            assert x_out == 2**bitsize - x_in


def test_negate_symbolic_classical_sim():
    bloq = _negate_symb()
    with pytest.raises(ValueError, match="Cannot simulate symbolic bloq"):
        bloq.on_classical_vals(x=1)
