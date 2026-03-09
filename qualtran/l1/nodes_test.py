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

from qualtran.l1.nodes import CArgNode, CObjectNode, LiteralNode, TupleNode


def test_literal_node():
    node = LiteralNode(1)
    assert node.canonical_str() == '1'

    node = LiteralNode(1.5)
    assert node.canonical_str() == '1.5'

    node = LiteralNode("hello")
    assert node.canonical_str() == "'hello'"


def test_tuple_node():
    # Single item
    node = TupleNode([LiteralNode(1)])
    assert node.canonical_str() == '(1,)'

    # Multiple items
    node = TupleNode([LiteralNode(1), LiteralNode(2)])
    assert node.canonical_str() == '(1, 2)'

    # Nested tuple
    node = TupleNode([LiteralNode(1), TupleNode([LiteralNode(2)])])
    assert node.canonical_str() == '(1, (2,))'


def test_carg_node():
    # Positional arg (no key)
    node = CArgNode(None, LiteralNode(1))
    assert node.canonical_str() == '1'

    # Keyword arg
    node = CArgNode('x', LiteralNode(1))
    assert node.canonical_str() == 'x=1'


def test_cobject_node():
    # No args
    node = CObjectNode('MyObj', [])
    assert node.canonical_str() == 'MyObj'

    # Positional args
    node = CObjectNode('MyObj', [CArgNode(None, LiteralNode(1))])
    assert node.canonical_str() == 'MyObj(1)'

    # Keyword args
    node = CObjectNode('MyObj', [CArgNode('x', LiteralNode(1))])
    assert node.canonical_str() == 'MyObj(x=1)'

    # Mixed args
    node = CObjectNode('MyObj', [CArgNode(None, LiteralNode(1)), CArgNode('y', LiteralNode(2))])
    assert node.canonical_str() == 'MyObj(1, y=2)'
