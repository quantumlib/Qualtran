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

from qualtran.bloqs.basic_gates import Toffoli
from qualtran.bloqs.factoring.mod_sub import ModNeg, ModSub


def test_mod_sub():
    bloq = ModSub(bitsize=8, p=3)
    assert bloq.short_name() == 'y = y - x mod 3'
    assert bloq.bloq_counts() == {(48, Toffoli())}


def test_mod_neg():
    bloq = ModNeg(bitsize=8, p=3)
    assert bloq.short_name() == 'x = -x mod 3'
    assert bloq.bloq_counts() == {(16, Toffoli())}
