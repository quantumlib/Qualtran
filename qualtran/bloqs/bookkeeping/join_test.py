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

from typing import Type, Union

import numpy as np
import pytest

from qualtran import BloqBuilder, QAny, Side
from qualtran.bloqs.bookkeeping import Allocate, Join, Split
from qualtran.bloqs.bookkeeping.join import _join


def test_join(bloq_autotester):
    bloq_autotester(_join)


@pytest.mark.parametrize('n', [5, 123])
@pytest.mark.parametrize('bloq_cls', [Split, Join])
def test_register_sizes_add_up(bloq_cls: Union[Type[Split], Type[Join]], n):
    bloq = bloq_cls(QAny(n))
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


def test_util_bloqs_tensor_contraction():
    bb = BloqBuilder()
    qs1 = bb.add(Allocate(QAny(10)))
    qs2 = bb.add(Split(QAny(10)), reg=qs1)
    qs3 = bb.add(Join(QAny(10)), reg=qs2)
    cbloq = bb.finalize(out=qs3)
    expected = np.zeros(2**10)
    expected[0] = 1
    np.testing.assert_allclose(cbloq.tensor_contract(), expected)
