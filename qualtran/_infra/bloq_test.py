#  Copyright 2023 Google LLC
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

from qualtran import CompositeBloq, DecomposeTypeError, Side
from qualtran.bloqs.for_testing import TestAtom, TestTwoBitOp
from qualtran.testing import execute_notebook

# Note: The `Bloq` abstract interface has many protocols and implementations. Each
# protocol and method is unit-tested within its own module or package.


def test_bloq():
    tb = TestTwoBitOp()
    assert len(tb.signature) == 2
    ctrl, trg = tb.signature
    assert ctrl.bitsize == 1
    assert ctrl.side == Side.THRU
    assert tb.pretty_name() == 'TestTwoBitOp'

    with pytest.raises(DecomposeTypeError):
        tb.decompose_bloq()


def test_as_composite_bloq():
    tb = TestAtom()
    assert not tb.supports_decompose_bloq()
    cb = tb.as_composite_bloq()
    assert isinstance(cb, CompositeBloq)
    bloqs = list(cb.bloq_instances)
    assert len(bloqs) == 1
    assert bloqs[0].bloq == tb

    cb2 = cb.as_composite_bloq()
    assert cb is cb2


def test_notebook():
    execute_notebook('Bloqs-Tutorial')
