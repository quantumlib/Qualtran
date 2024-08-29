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
from qualtran import BloqBuilder
from qualtran.bloqs.arithmetic.conversions.contiguous_index import (
    _to_contg_index,
    ToContiguousIndex,
)


def test_to_contigous_index(bloq_autotester):
    bloq_autotester(_to_contg_index)


def test_to_contiguous_index_t_complexity():
    bb = BloqBuilder()
    bitsize = 5
    q0 = bb.add_register('mu', bitsize)
    q1 = bb.add_register('nu', bitsize)
    out = bb.add_register('s', 2 * bitsize)
    q0, q1, out = bb.add(ToContiguousIndex(bitsize, 2 * bitsize), mu=q0, nu=q1, s=out)
    cbloq = bb.finalize(mu=q0, nu=q1, s=out)
    assert cbloq.t_complexity().t == 4 * 29
