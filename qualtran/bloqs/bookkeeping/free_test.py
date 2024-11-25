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

from qualtran import BloqBuilder, QAny, Soquet
from qualtran.bloqs.bookkeeping import Allocate, Free, Join, Split
from qualtran.bloqs.bookkeeping.free import _free


def test_free(bloq_autotester):
    bloq_autotester(_free)


def test_util_bloqs():
    bb = BloqBuilder()
    qs1 = bb.add(Allocate(QAny(10)))
    assert isinstance(qs1, Soquet)
    qs2 = bb.add(Split(QAny(10)), reg=qs1)
    assert qs2.shape == (10,)
    qs3 = bb.add(Join(QAny(10)), reg=qs2)
    assert isinstance(qs3, Soquet)
    no_return = bb.add(Free(QAny(10)), reg=qs3)
    assert no_return is None
    assert bb.finalize().tensor_contract() == 1.0
