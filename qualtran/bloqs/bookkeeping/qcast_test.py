#  Copyright 2026 Google LLC
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

import qualtran as qlt
import qualtran.dtype as qdt
from qualtran.bloqs.bookkeeping.qcast import QCast


def test_qcast_is_bookkeeping():
    from qualtran.bloqs.bookkeeping._bookkeeping_bloq import _BookkeepingBloq

    sig = qlt.Signature([qlt.Register('reg', qdt.QUInt(4))])
    bloq = QCast(signature=sig)
    assert isinstance(bloq, _BookkeepingBloq)


def test_qcast_signature():
    sig = qlt.Signature([qlt.Register('reg', qdt.QBit())])
    bloq = QCast(signature=sig)
    assert bloq.signature is sig


def test_qcast_decompose_raises():
    sig = qlt.Signature([qlt.Register('x', qdt.QBit())])
    bloq = QCast(signature=sig)
    with pytest.raises(qlt.DecomposeTypeError, match='is atomic'):
        bloq.decompose_bloq()


def test_qcast_equality():
    sig1 = qlt.Signature([qlt.Register('x', qdt.QBit())])
    sig2 = qlt.Signature([qlt.Register('x', qdt.QBit())])
    sig3 = qlt.Signature([qlt.Register('y', qdt.QUInt(4))])
    assert QCast(signature=sig1) == QCast(signature=sig2)
    assert QCast(signature=sig1) != QCast(signature=sig3)
