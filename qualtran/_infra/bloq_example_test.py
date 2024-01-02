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

from qualtran import bloq_example, BloqExample
from qualtran.bloqs.for_testing import TestAtom


def _tester_bloq_func() -> TestAtom:
    return TestAtom()


@bloq_example
def _tester_bloq() -> TestAtom:
    return TestAtom()


def test_bloq_example_explicit():
    be = BloqExample(func=_tester_bloq_func, name='tester_bloq', bloq_cls=TestAtom)
    assert be.name == 'tester_bloq'
    assert be.bloq_cls == TestAtom


def test_bloq_example_decorator():
    be = _tester_bloq
    assert be.name == 'tester_bloq'
    assert be.bloq_cls == TestAtom
