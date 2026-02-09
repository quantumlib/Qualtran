#  Copyright 2025 Google LLC
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

from qualtran.l1._eval import eval_cvalue_node, UnevaluatedCValue
from qualtran.l1.nodes import CArgNode, CObjectNode, LiteralNode


def test_safe_eval_prevents_arbitrary_code():
    # Try to execute os.system('echo hacked')
    # serialized as CObjectNode(name='os.system', cargs=[CArgNode(None, LiteralNode('echo hacked'))])

    node = CObjectNode(name='os.system', cargs=[CArgNode(None, LiteralNode('echo hacked'))])

    # With safe=True, it should return UnevaluatedCValue
    result = eval_cvalue_node(node, safe=True)
    assert isinstance(result, UnevaluatedCValue)
    assert result.name == 'os.system'


def test_safe_eval_allows_whitelisted():
    # Register is whitelisted
    # We need valid args for Register

    # We need to construct a node that evaluates to QBit() first
    qbit_node = CObjectNode(name='QBit', cargs=[])

    node = CObjectNode(
        name='Register', cargs=[CArgNode('name', LiteralNode('q')), CArgNode('dtype', qbit_node)]
    )

    result = eval_cvalue_node(node, safe=True)
    from qualtran import Register

    assert isinstance(result, Register)
    assert result.name == 'q'


def test_safe_eval_symbol_enforces_string():
    # Symbol with non-string arg should fail in safe mode
    node = CObjectNode(name='Symbol', cargs=[CArgNode(None, LiteralNode(123))])

    with pytest.raises(TypeError, match="Symbol nodes must take one string argument"):
        eval_cvalue_node(node, safe=True)


def test_safe_eval_propagates_safe_flag():
    # Nested object that is not whitelisted
    inner_node = CObjectNode(name='os.system', cargs=[CArgNode(None, LiteralNode('echo hacked'))])

    # Tuple containing the unsafe node
    from qualtran.l1.nodes import TupleNode

    tuple_node = TupleNode([inner_node])

    # Should return tuple of UnevaluatedCValue
    result = eval_cvalue_node(tuple_node, safe=True)
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert isinstance(result[0], UnevaluatedCValue)
    assert result[0].name == 'os.system'
