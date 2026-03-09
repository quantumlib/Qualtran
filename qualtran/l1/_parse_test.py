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

import io
import sys
from typing import Any

import pytest

from qualtran.l1._parse import (
    dump_ast,
    l1_ast_node_to_json,
    parse_objectstring,
    QualtranL1Parser,
    Token,
    tokenize,
)
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


def test_parse_newlines_comments():
    code = "MyBloq(\n# A comment\n1)"
    tokens = tokenize(code)
    # verify it skips comments and newlines correctly
    node = QualtranL1Parser(tokens).parse_cobject_node()
    assert isinstance(node, CObjectNode)
    assert len(node.cargs) == 1


def test_parse_mismatch():
    with pytest.raises(ValueError, match="unexpected on line"):
        tokenize("MyBloq(1, ^)")


def test_parse_prev():
    tokens = tokenize("MyBloq(1)")
    parser = QualtranL1Parser(tokens)
    parser.advance()
    assert parser.prev().type == 'NAME'


def test_parse_missing_delimiter():
    tokens = tokenize("MyBloq(1 2)")
    parser = QualtranL1Parser(tokens)
    with pytest.raises(ValueError, match="Extraneous elements"):
        parser.parse_cobject_node()


def test_parse_bloq_key():
    tokens = tokenize("MyBloq()")
    parser = QualtranL1Parser(tokens)
    assert parser.parse_bloq_key() == "MyBloq"


def test_parse_empty_tuple():
    tokens = tokenize("()")
    parser = QualtranL1Parser(tokens)
    node = parser.parse_cvalue()
    assert isinstance(node, TupleNode)
    assert len(node.items) == 0


def test_parse_float_literal():
    tokens = tokenize("1.5")
    parser = QualtranL1Parser(tokens)
    node = parser.parse_cvalue()
    assert isinstance(node, LiteralNode)
    assert node.value == 1.5

    tokens2 = tokenize("1e-3")
    parser2 = QualtranL1Parser(tokens2)
    node2 = parser2.parse_cvalue()
    assert isinstance(node2, LiteralNode)
    assert node2.value == 1e-3


def test_ast_node_to_json():

    node = CArgNode('x', LiteralNode(1))
    f = io.StringIO()
    dump_ast(node, f)
    result = f.getvalue()
    assert '"_l1_node": "CArgNode"' in result
    assert '"key": "x"' in result

    # Test fallback
    assert l1_ast_node_to_json("string") == "string"


def test_parse_int_literal_invalid():
    parser = QualtranL1Parser([Token('NUMBER', '1.5', 1, 0)])
    with pytest.raises(ValueError, match="Expected an integer literal"):
        parser.parse_int_literal()


def test_parse_cvalue_cobject():
    parser = QualtranL1Parser([Token('NAME', 'MyBloq', 1, 0), Token('EOF', '', 1, 0)])
    node = parser.parse_cvalue()
    assert isinstance(node, CObjectNode)
    assert node.name == 'MyBloq'


def test_parse_nested():
    from qualtran.l1._eval import eval_cvalue_node

    n = 100
    s = '(' * n + '5,' + ')' * n
    tokens = tokenize(s)
    parser = QualtranL1Parser(tokens)
    ast = parser.parse_cvalue()
    result = eval_cvalue_node(ast)

    should_be: Any = 5
    for i in range(n):
        should_be = (should_be,)
    assert result == should_be


def test_parse_too_nested():
    n = sys.getrecursionlimit() + 1
    s = '(' * n + '5,' + ')' * n
    tokens = tokenize(s)
    parser = QualtranL1Parser(tokens)
    with pytest.raises(RecursionError):
        ast = parser.parse_cvalue()


def test_parse_module_empty():
    parser = QualtranL1Parser([Token('EOF', '', 1, 0)])
    module = parser.parse_module()
    assert len(module.qdefs) == 0


from qualtran.l1._parse import parse_module


def test_parse_module_full():
    code = """
extern qdef MyBloq() [
    q: t -> |
]

qdef OtherBloq() from qualtran.bloqs.OtherBloq() [
    q: t
] {
    q = MyBloq()[q=q]
    return [q=q]
}
"""
    module = parse_module(code)
    assert len(module.qdefs) == 2
    assert module.qdefs[0].bloq_key == "MyBloq"
    assert module.qdefs[0].qsignature[0].name == "q"
    assert module.qdefs[1].bloq_key == "OtherBloq"
    assert module.qdefs[1].body[0].lvalues == ("q",)
    assert module.qdefs[1].body[0].bloq_key == "MyBloq"


def test_parse_lvalues_pipe():
    parser = QualtranL1Parser(tokenize("| = MyBloq()"))
    lvals = parser.parse_lvalues()
    assert lvals == []


def test_parse_qarg_nested():
    parser = QualtranL1Parser(tokenize("target=[x[0], [y[1], z[2]]] ]"))
    qarg = parser.parse_qarg()
    assert qarg.key == "target"
    assert len(qarg.value) == 2
    assert qarg.value[0].name == "x"
    assert qarg.value[0].idx == (0,)
    assert len(qarg.value[1]) == 2


def test_parse_signature_types():
    code = "qdef Foo() [ a: t, b: t -> |, c: | -> t, d: t1 -> t2, e: t1[2, 3] ] {}"
    module = parse_module(code)
    qdef = module.qdefs[0]
    assert len(qdef.qsignature) == 5
    assert qdef.qsignature[0].name == "a"
    assert qdef.qsignature[1].name == "b"
    assert qdef.qsignature[2].name == "c"
    assert qdef.qsignature[3].name == "d"
    assert qdef.qsignature[4].name == "e"
    assert qdef.qsignature[4].dtype.shape == [2, 3]


def test_parse_errors():
    with pytest.raises(ValueError, match="Expected 'extern qdef'"):
        parse_module("extern foo MyBloq() [q: t -> |]")

    with pytest.raises(ValueError, match="qdef from must start with 'from'"):
        parser = QualtranL1Parser(tokenize("foo qualtran.bloqs.Bloq()"))
        parser.parse_qdef_from()

    with pytest.raises(ValueError, match="return statement must start with 'return'"):
        parser = QualtranL1Parser(tokenize("foo [a=b]"))
        parser.parse_return_statement()

    with pytest.raises(ValueError, match="only one lvalue may be specified"):
        parse_module("qdef Foo() [] { a, b = qualtran.bloqs.Bloq }")


def test_parse_empty_brackets():
    code = "qdef Foo() [] { a = MyBloq()[] } qdef Bar() [a: t[]] { c = qualtran.bloqs.Bloq }"
    module = parse_module(code)
    assert module.qdefs[0].qsignature == ()
    assert module.qdefs[0].body[0].qargs == ()
    assert module.qdefs[1].qsignature[0].dtype.shape == []
    assert module.qdefs[1].body[0].alias == "c"
