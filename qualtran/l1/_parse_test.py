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

from qualtran.l1._parse import parse_objectstring, QualtranL1Parser, tokenize
from qualtran.l1.nodes import CArgNode, CObjectNode, LiteralNode, TupleNode


def test_tokenize():
    code = "MyBloq(1, x='foo')"
    tokens = tokenize(code)
    assert len(tokens) > 0
    assert tokens[0].type == 'NAME'
    assert tokens[0].value == 'MyBloq'
    assert tokens[1].type == 'LPAREN'
    assert tokens[2].type == 'NUMBER'
    assert tokens[2].value == '1'
    assert tokens[-1].type == 'EOF'


def test_parse_qualified_identifier():
    tokens = tokenize("qualtran.bloqs.MyBloq")
    parser = QualtranL1Parser(tokens)
    name = parser.parse_qualified_identifier()
    assert name == "qualtran.bloqs.MyBloq"


def test_parse_cobject_node_no_args():
    tokens = tokenize("MyBloq")
    parser = QualtranL1Parser(tokens)
    node = parser.parse_cobject_node()
    assert isinstance(node, CObjectNode)
    assert node.name == "MyBloq"
    assert node.cargs == ()


def test_parse_cobject_node_empty_args():
    tokens = tokenize("MyBloq()")
    parser = QualtranL1Parser(tokens)
    node = parser.parse_cobject_node()
    assert isinstance(node, CObjectNode)
    assert node.name == "MyBloq"
    assert node.cargs == ()


def test_parse_cobject_node_with_args():
    tokens = tokenize("MyBloq(1, x='foo')")
    parser = QualtranL1Parser(tokens)
    node = parser.parse_cobject_node()
    assert isinstance(node, CObjectNode)
    assert node.name == "MyBloq"
    assert len(node.cargs) == 2

    arg0 = node.cargs[0]
    assert isinstance(arg0, CArgNode)
    assert arg0.key is None
    assert isinstance(arg0.value, LiteralNode)
    assert arg0.value.value == 1

    arg1 = node.cargs[1]
    assert isinstance(arg1, CArgNode)
    assert arg1.key == 'x'
    assert isinstance(arg1.value, LiteralNode)
    assert arg1.value.value == 'foo'


def test_parse_cvalue_tuple():
    tokens = tokenize("(1, 2)")
    parser = QualtranL1Parser(tokens)
    node = parser.parse_cvalue()
    assert isinstance(node, TupleNode)
    assert len(node.items) == 2
    assert isinstance(node.items[0], LiteralNode)
    assert node.items[0].value == 1
    assert isinstance(node.items[1], LiteralNode)
    assert node.items[1].value == 2


def test_parse_cvalue_nested_tuple():
    tokens = tokenize("(1, (2, 3))")
    parser = QualtranL1Parser(tokens)
    node = parser.parse_cvalue()
    assert isinstance(node, TupleNode)
    assert len(node.items) == 2
    assert isinstance(node.items[0], LiteralNode)
    assert node.items[0].value == 1
    assert isinstance(node.items[1], TupleNode)
    assert isinstance(node.items[1].items[0], LiteralNode)
    assert node.items[1].items[0].value == 2


def test_parse_objectstring():
    code = "MyBloq(1, x='foo')"
    node = parse_objectstring(code)
    assert isinstance(node, CObjectNode)
    assert node.name == "MyBloq"


def test_parse_error_unexpected_token():
    tokens = tokenize("MyBloq(1, =)")
    parser = QualtranL1Parser(tokens)
    with pytest.raises(ValueError, match='Unexpected token'):
        parser.parse_cobject_node()


def test_parse_int_literal():
    tokens = tokenize("123")
    parser = QualtranL1Parser(tokens)
    val = parser.parse_int_literal()
    assert val == 123

    tokens = tokenize("'foo'")
    parser = QualtranL1Parser(tokens)
    with pytest.raises(ValueError, match='Expected an integer.*foo'):
        parser.parse_int_literal()
