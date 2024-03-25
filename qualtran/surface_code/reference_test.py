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

import pytest

from qualtran.surface_code import Reference


@pytest.mark.parametrize(
    "reference, expected",
    [
        (Reference(), 'Reference()'),
        (Reference(url='xyz'), "Reference(url='xyz')"),
        (Reference(page=1), 'Reference(page=1)'),
        (Reference(comment='xyz'), "Reference(comment='xyz')"),
        (Reference(url='a', page=3), "Reference(url='a', page=3)"),
        (Reference(url='a', page=5, comment='xyz'), "Reference(url='a', page=5, comment='xyz')"),
    ],
)
def test_representation(reference: Reference, expected: str):
    assert repr(reference) == expected
