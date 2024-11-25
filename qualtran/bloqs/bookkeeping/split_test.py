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

import numpy as np
import pytest
import sympy

from qualtran import BloqBuilder, QAny, QUInt
from qualtran.bloqs.basic_gates import XGate
from qualtran.bloqs.bookkeeping import Split
from qualtran.bloqs.bookkeeping.split import _split
from qualtran.simulation.classical_sim import call_cbloq_classically


def test_split(bloq_autotester):
    bloq_autotester(_split)


def test_no_symbolic():
    n = sympy.Symbol('n')
    with pytest.raises(ValueError, match=r'.*cannot have a symbolic data type\.'):
        Split(QUInt(n))


def test_classical_sim():
    bb = BloqBuilder()
    x = bb.allocate(4)
    xs = bb.split(x)
    xs_1_orig = xs[1]  # keep a copy for later
    xs[1] = bb.add(XGate(), q=xs[1])
    y = bb.join(xs)
    cbloq = bb.finalize(y=y)

    ret, assign = call_cbloq_classically(cbloq.signature, vals={}, binst_graph=cbloq._binst_graph)
    assert assign[x] == 0

    assert assign[xs[0]] == 0
    assert assign[xs_1_orig] == 0
    assert assign[xs[2]] == 0
    assert assign[xs[3]] == 0

    assert assign[xs[1]] == 1
    assert assign[y] == 4

    assert ret == {'y': 4}


def test_classical_sim_dtypes():
    s = Split(QAny(8))
    (xx,) = s.call_classically(reg=255)
    assert isinstance(xx, np.ndarray)
    assert xx.tolist() == [1, 1, 1, 1, 1, 1, 1, 1]

    with pytest.raises(ValueError):
        _ = s.call_classically(reg=256)

    # with numpy types
    (xx,) = s.call_classically(reg=np.uint8(255))
    assert isinstance(xx, np.ndarray)
    assert xx.tolist() == [1, 1, 1, 1, 1, 1, 1, 1]

    # Warning: numpy will wrap too-large values
    (xx,) = s.call_classically(reg=np.uint8(256))
    assert isinstance(xx, np.ndarray)
    assert xx.tolist() == [0, 0, 0, 0, 0, 0, 0, 0]

    with pytest.raises(ValueError):
        _ = s.call_classically(reg=np.uint16(256))
