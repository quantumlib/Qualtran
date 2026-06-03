#  Copyright 2026 Google LLC
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

import qualtran as qlt
import qualtran.dtype as qdt

from qualtran.l1._dtypes import get_builtin_qdtype_mapping, reg_to_qdtype_node
from qualtran.l1._eval import eval_qdtype_node


@pytest.mark.parametrize(
    'reg',
    [
        qlt.Register('x', qdt.QBit()),
        qlt.Register('x', qdt.QAny(10)),
        qlt.Register('x', qdt.QUInt(10)),
        qlt.Register('x', qdt.QInt(10)),
        qlt.Register('x', qdt.QFxp(10, 5)),
        qlt.Register('x', qdt.QMontgomeryUInt(10)),
        qlt.Register('x', qdt.BQUInt(10, 5)),
        qlt.Register('x', qdt.CBit()),
        qlt.Register('x', qdt.QIntOnesComp(10)),
        qlt.Register('x', qdt.QIntSignMag(10)),
    ],
)
def test_qdtype_roundtrip(reg):
    """Test that dtype serialization through AST nodes roundtrips correctly."""
    qdtype_node = reg_to_qdtype_node(reg)
    roundtripped, shape = eval_qdtype_node(qdtype_node)
    assert roundtripped == reg.dtype
    assert shape == ()


def test_builtin_mapping_contains_all_expected():
    """Verify all expected types are in the builtin mapping."""
    mapping = get_builtin_qdtype_mapping()
    expected_names = [
        'BQUInt',
        'QAny',
        'QBit',
        'QInt',
        'QUInt',
        'QFxp',
        'QMontgomeryUInt',
        'CBit',
        'QIntOnesComp',
        'QIntSignMag',
    ]
    for name in expected_names:
        assert name in mapping, f'{name} missing from builtin qdtype mapping'


@pytest.mark.parametrize(
    ('objectstring', 'expected'),
    [
        ('QIntOnesComp(8)', qdt.QIntOnesComp(8)),
        ('QIntSignMag(8)', qdt.QIntSignMag(8)),
        ('QInt(8)', qdt.QInt(8)),
        ('QUInt(8)', qdt.QUInt(8)),
    ],
)
def test_load_objectstring_safe(objectstring, expected):
    """Builtin dtypes can be loaded from objectstrings with safe=True."""
    from qualtran.l1 import load_objectstring

    obj = load_objectstring(objectstring)
    assert obj == expected

