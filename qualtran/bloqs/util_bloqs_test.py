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

from typing import Type

import numpy as np
import pytest

from qualtran import Bloq, BloqBuilder, Side, Soquet
from qualtran.bloqs.basic_gates import XGate
from qualtran.bloqs.util_bloqs import Allocate, Free, Join, Split
from qualtran.simulation.classical_sim import _cbloq_call_classically
from qualtran.testing import execute_notebook


@pytest.mark.parametrize('n', [5, 123])
@pytest.mark.parametrize('bloq_cls', [Split, Join])
def test_register_sizes_add_up(bloq_cls: Type[Bloq], n):
    bloq = bloq_cls(n)
    for name, group_regs in bloq.signature.groups():

        if any(reg.side is Side.THRU for reg in group_regs):
            assert not any(reg.side != Side.THRU for reg in group_regs)
            continue

        lefts = [reg for reg in group_regs if reg.side & Side.LEFT]
        left_size = np.prod([l.total_bits() for l in lefts])
        rights = [reg for reg in group_regs if reg.side & Side.RIGHT]
        right_size = np.prod([r.total_bits() for r in rights])

        assert left_size > 0
        assert left_size == right_size


def test_util_bloqs():
    bb = BloqBuilder()
    qs1 = bb.add(Allocate(10))
    assert isinstance(qs1, Soquet)
    qs2 = bb.add(Split(10), split=qs1)
    assert qs2.shape == (10,)
    qs3 = bb.add(Join(10), join=qs2)
    assert isinstance(qs3, Soquet)
    no_return = bb.add(Free(10), free=qs3)
    assert no_return is None


def test_classical_sim():
    bb = BloqBuilder()
    x = bb.allocate(4)
    xs = bb.split(x)
    xs_1_orig = xs[1]  # keep a copy for later
    xs[1] = bb.add(XGate(), q=xs[1])
    y = bb.join(xs)
    cbloq = bb.finalize(y=y)

    ret, assign = _cbloq_call_classically(cbloq.signature, vals={}, binst_graph=cbloq._binst_graph)
    assert assign[x] == 0

    assert assign[xs[0]] == 0
    assert assign[xs_1_orig] == 0
    assert assign[xs[2]] == 0
    assert assign[xs[3]] == 0

    assert assign[xs[1]] == 1
    assert assign[y] == 4

    assert ret == {'y': 4}


def test_classical_sim_dtypes():
    s = Split(n=8)
    (xx,) = s.call_classically(split=255)
    assert xx.tolist() == [1, 1, 1, 1, 1, 1, 1, 1]

    with pytest.raises(ValueError):
        _ = s.call_classically(split=256)

    # with numpy types
    (xx,) = s.call_classically(split=np.uint8(255))
    assert xx.tolist() == [1, 1, 1, 1, 1, 1, 1, 1]

    # Warning: numpy will wrap too-large values
    (xx,) = s.call_classically(split=np.uint8(256))
    assert xx.tolist() == [0, 0, 0, 0, 0, 0, 0, 0]

    with pytest.raises(ValueError):
        _ = s.call_classically(split=np.uint16(256))


def test_notebook():
    execute_notebook('util_bloqs')
