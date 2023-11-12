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

from qualtran.bloqs.factoring.mod_add import CtrlModAddK, CtrlScaleModAdd, ModAdd
from qualtran.resource_counting import SympySymbolAllocator
from qualtran.bloqs.basic_gates import Toffoli


def test_ctrl_scale_mod_add():
    bloq = CtrlScaleModAdd(k=123, mod=13 * 17, bitsize=8)
    assert bloq.short_name() == 'y += x*123 % 221'

    counts = bloq.bloq_counts()
    ((bloq, n),) = counts.items()
    assert n == 8


def test_ctrl_mod_add_k():
    bloq = CtrlModAddK(k=123, mod=13 * 17, bitsize=8)
    assert bloq.short_name() == 'x += 123 % 221'

    counts = bloq.bloq_counts()
    ((bloq, n),) = counts.items()
    assert n == 5


def test_mod_add():
    bloq = ModAdd(bitsize=8, p=3)
    assert bloq.short_name() == 'y = y + x mod 3'
    assert bloq.bloq_counts() == {(32, Toffoli())}
