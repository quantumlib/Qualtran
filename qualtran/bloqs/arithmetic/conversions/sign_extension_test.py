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

from qualtran import BloqBuilder, QInt, QIntOnesComp, QUInt
from qualtran.bloqs.arithmetic.conversions.sign_extension import (
    _sign_extend,
    _sign_truncate,
    SignExtend,
    SignTruncate,
)
from qualtran.bloqs.basic_gates import IntEffect, IntState


def test_examples(bloq_autotester):
    bloq_autotester(_sign_extend)
    bloq_autotester(_sign_truncate)


@pytest.mark.parametrize("inp, out", [(QInt(8), QInt(16)), (QIntOnesComp(8), QIntOnesComp(16))])
def test_adjoint(inp, out):
    extend_bloq = SignExtend(inp, out)
    trunc_bloq = SignTruncate(out, inp)
    assert extend_bloq.adjoint() == trunc_bloq
    assert trunc_bloq.adjoint() == extend_bloq


@pytest.mark.parametrize(
    "inp, out", [(QInt(8), QInt(8)), (QInt(8), QInt(7)), (QInt(8), QIntOnesComp(12))]
)
def test_sign_extend_raises_on_incompatible_dtypes(inp, out):
    with pytest.raises(ValueError):
        SignExtend(inp, out)


@pytest.mark.parametrize(
    "inp, out", [(QInt(8), QInt(8)), (QInt(8), QInt(7)), (QInt(8), QIntOnesComp(12))]
)
def test_sign_truncate_raises_on_incompatible_dtypes(inp, out):
    with pytest.raises(ValueError):
        SignTruncate(out, inp)


def _as_unsigned(num: int, bitsize: int):
    # TODO remove this once IntState supports signed values
    return QUInt(bitsize).from_bits(QInt(bitsize).to_bits(num))


@pytest.mark.parametrize("l, r", [(2, 4)])
def test_sign_extend_tensor(l: int, r: int):
    bloq = SignExtend(QInt(l), QInt(r))

    for x in QInt(l).get_classical_domain():
        bb = BloqBuilder()
        qx = bb.add(IntState(_as_unsigned(x, l), l))
        qx = bb.add(bloq, x=qx)
        bb.add(IntEffect(_as_unsigned(x, r), r), val=qx)
        cbloq = bb.finalize()

        np.testing.assert_allclose(cbloq.tensor_contract(), 1)


@pytest.mark.parametrize("l, r", [(2, 4)])
def test_sign_extend_classical_sim(l: int, r: int):
    bloq = SignExtend(QInt(l), QInt(r))

    for x in QInt(l).get_classical_domain():
        (y,) = bloq.call_classically(x=x)
        assert y == x


@pytest.mark.parametrize("l, r", [(4, 2)])
def test_sign_truncate_tensor(l: int, r: int):
    bloq = SignTruncate(QInt(l), QInt(r))

    for x in QInt(r).get_classical_domain():
        bb = BloqBuilder()
        qx = bb.add(IntState(_as_unsigned(x, l), l))
        qx = bb.add(bloq, x=qx)
        bb.add(IntEffect(_as_unsigned(x, r), r), val=qx)
        cbloq = bb.finalize()

        np.testing.assert_allclose(cbloq.tensor_contract(), 1)


@pytest.mark.parametrize("l, r", [(4, 2)])
def test_sign_truncate_classical_sim(l: int, r: int):
    bloq = SignTruncate(QInt(l), QInt(r))

    for x in QInt(r).get_classical_domain():
        (y,) = bloq.call_classically(x=x)
        assert y == x


def test_sign_truncate_raises_on_invalid_truncation_bits():
    bloq = SignTruncate(QInt(4), QInt(2))
    with pytest.raises(ValueError):
        # 1100 is invalid as the new sign bit is 0, which doesn't match the dropped bits 11.
        bloq.call_classically(x=0b1100)
