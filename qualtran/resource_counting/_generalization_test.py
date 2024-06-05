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
from typing import Optional

from qualtran import Bloq
from qualtran.bloqs.for_testing import TestAtom
from qualtran.resource_counting._generalization import _make_composite_generalizer


def test_make_composite_generalizer():
    def func1(b: Bloq) -> Optional[Bloq]:
        if isinstance(b, TestAtom):
            return TestAtom()
        return b

    def func2(b: Bloq) -> Optional[Bloq]:
        if isinstance(b, TestAtom):
            return None
        return b

    b = TestAtom(tag='test')
    assert func1(b) == TestAtom()
    assert func2(b) is None

    g00 = _make_composite_generalizer()
    g10 = _make_composite_generalizer(func1)
    g01 = _make_composite_generalizer(func2)
    g11 = _make_composite_generalizer(func1, func2)
    g11_r = _make_composite_generalizer(func2, func1)

    assert g00(b) == b
    assert g10(b) == TestAtom()
    assert g01(b) is None
    assert g11(b) is None
    assert g11_r(b) is None
