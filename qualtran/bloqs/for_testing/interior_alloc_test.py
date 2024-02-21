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
import qualtran.testing as qlt_testing
from qualtran import QAny
from qualtran.bloqs.basic_gates.swap import Swap
from qualtran.bloqs.for_testing.interior_alloc import InteriorAlloc
from qualtran.bloqs.util_bloqs import Allocate, Free


def test_interior_alloc():
    ia = InteriorAlloc(10)
    qlt_testing.assert_valid_bloq_decomposition(ia)
    g, counts = ia.call_graph(max_depth=1)
    assert counts == {Allocate(QAny(10)): 1, Free(QAny(10)): 1, Swap(10): 2}
