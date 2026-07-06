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
    parse_module,
    parse_objectstring,
    QualtranL1Parser,
    Token,
    tokenize,
)
from qualtran.l1.nodes import (
    CArgNode,
    CObjectNode,
    LiteralNode,
    QArgValueNode,
    QCallNode,
    QCastNode,
    QDefImplNode,
    QDTypeNode,
    QReturnNode,
    TupleNode,
)


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
        _ast = parser.parse_cvalue()


def test_parse_module_empty():
    parser = QualtranL1Parser([Token('EOF', '', 1, 0)])
    module = parser.parse_module()
    assert len(module.qdefs) == 0


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
    assert isinstance(module.qdefs[1], QDefImplNode)
    stmt0 = module.qdefs[1].body[0]
    assert isinstance(stmt0, QCallNode)
    assert len(stmt0.lvalues) == 1
    assert stmt0.lvalues[0].name == "q"
    assert stmt0.bloq_key == "MyBloq"


def test_parse_lvalues_pipe():
    parser = QualtranL1Parser(tokenize("| = MyBloq()"))
    lvals = parser.parse_lvalues()
    assert lvals == []


def test_parse_qarg_nested():
    parser = QualtranL1Parser(tokenize("target=[x[0], [y[1], z[2]]] ]"))
    qarg = parser.parse_qarg()
    assert qarg.key == "target"
    assert isinstance(qarg.value, tuple) or isinstance(qarg.value, list)
    assert len(qarg.value) == 2
    assert isinstance(qarg.value[0], QArgValueNode)
    assert qarg.value[0].name == "x"
    assert qarg.value[0].idx == (0,)
    assert isinstance(qarg.value[1], tuple) or isinstance(qarg.value[1], list)
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
    assert isinstance(qdef.qsignature[4].dtype, QDTypeNode)
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
    assert module.qdefs[0].body[0].qargs == ()  # type: ignore
    assert module.qdefs[1].qsignature[0].dtype.shape == []  # type: ignore
    assert module.qdefs[1].body[0].alias == "c"  # type: ignore[attr-defined]


def test_parse_annotations():
    code = """# Qualtran-L1
# 1.0.0

qdef BitwiseNot(QAny(4))
from BitwiseNot(QAny(4))
[
    x: QAny(4) @ (3, 0),
] {
    reg @ (1, 2, 3, 4, 2, 3, 4) = Split(QAny(4)) @ (5, 0) [reg=x]
    q @ 1                       = X @ (8, 0, False)       [q=reg[0] @ 1]
    q2 @ 2                      = X @ (9, 0, False)       [q=reg[1] @ 2]
    q3 @ 3                      = X @ (8, 0, False)       [q=reg[2] @ 3]
    q4 @ 4                      = X @ (9, 0, False)       [q=reg[3] @ 4]
    reg @ 0                     = Join(QAny(4)) @ (12, 0) [reg=[q, q2, q3, q4] @ (1, 2, 3, 4)]
                                  return                  [x=reg @ (16, 0)]
}

extern qdef Split(QAny(4))
from qualtran.bloqs.bookkeeping.Split(QAny(4))
[reg: QAny(4) -> QBit[4]]

extern qdef X
from qualtran.bloqs.basic_gates.XGate
[q: QBit @ oplus]

extern qdef Join(QAny(4))
from qualtran.bloqs.bookkeeping.Join(QAny(4))
[reg: QBit[4] -> QAny(4)]
"""
    module = parse_module(code)
    assert len(module.qdefs) == 4

    # Check BitwiseNot
    qdef = module.qdefs[0]
    assert qdef.bloq_key == "BitwiseNot(QAny(4))"
    assert len(qdef.qsignature) == 1
    sig = qdef.qsignature[0]
    assert sig.name == "x"
    assert sig.annotation is not None
    assert isinstance(sig.annotation, TupleNode)
    assert len(sig.annotation.items) == 2

    # Check body statements
    assert isinstance(qdef, QDefImplNode)
    assert len(qdef.body) == 7

    # Split call
    stmt0 = qdef.body[0]
    assert isinstance(stmt0, QCallNode)
    assert stmt0.bloq_key == "Split(QAny(4))"
    assert len(stmt0.lvalues) == 1
    assert stmt0.lvalues[0].name == "reg"
    assert stmt0.lvalues[0].annotation is not None
    assert stmt0.annotation is not None

    # X call
    stmt1 = qdef.body[1]
    assert isinstance(stmt1, QCallNode)
    assert stmt1.bloq_key == "X"
    assert stmt1.lvalues[0].name == "q"
    assert stmt1.lvalues[0].annotation is not None
    assert stmt1.annotation is not None
    assert stmt1.qargs[0].annotation is not None

    # Join call
    stmt5 = qdef.body[5]
    assert isinstance(stmt5, QCallNode)
    assert stmt5.bloq_key == "Join(QAny(4))"

    # Return
    stmt6 = qdef.body[6]
    assert isinstance(stmt6, QReturnNode)
    assert stmt6.ret_mapping[0].annotation is not None


def test_parse_annotation():
    parser = QualtranL1Parser(tokenize("@ 123"))
    annotation = parser.parse_annotation()
    assert isinstance(annotation, LiteralNode)
    assert annotation.value == 123

    parser = QualtranL1Parser(tokenize("not_an_annotation"))
    annotation = parser.parse_annotation()
    assert annotation is None


def test_parse_lvalues_with_annotation():
    parser = QualtranL1Parser(tokenize("a @ 1, b @ 2 ="))
    lvals = parser.parse_lvalues()
    assert len(lvals) == 2
    assert lvals[0].name == "a"
    assert lvals[0].annotation is not None
    assert isinstance(lvals[0].annotation, LiteralNode)
    assert lvals[0].annotation.value == 1
    assert lvals[1].name == "b"
    assert lvals[1].annotation is not None
    assert isinstance(lvals[1].annotation, LiteralNode)
    assert lvals[1].annotation.value == 2


def test_parse_qarg_with_annotation():
    parser = QualtranL1Parser(tokenize("ctrl=qvar @ 1"))
    qarg = parser.parse_qarg()
    assert qarg.key == "ctrl"
    assert qarg.annotation is not None
    assert isinstance(qarg.annotation, LiteralNode)
    assert qarg.annotation.value == 1

    # Test array indexing on key too
    parser = QualtranL1Parser(tokenize("reg[0]=qvar @ 2"))
    qarg = parser.parse_qarg()
    assert qarg.key == "reg[0]"
    assert qarg.annotation is not None
    assert isinstance(qarg.annotation, LiteralNode)
    assert qarg.annotation.value == 2


def test_parse_signature_with_annotation():
    code = "qdef Foo() [ a: t @ 1 ] {}"
    module = parse_module(code)
    qdef = module.qdefs[0]
    assert qdef.qsignature[0].annotation is not None
    assert isinstance(qdef.qsignature[0].annotation, LiteralNode)
    assert qdef.qsignature[0].annotation.value == 1


def test_parse_qcast():
    code = """# Qualtran-L1
# 1.0.0

qcast Split(QUInt(4))
[reg: QUInt(4) -> QBit[4]]
"""
    module = parse_module(code)
    assert len(module.qdefs) == 1
    qdef = module.qdefs[0]
    assert isinstance(qdef, QCastNode)
    assert qdef.bloq_key == "Split(QUInt(4))"
    assert len(qdef.qsignature) == 1
    assert qdef.cobject_from is None

    # Check the signature entry
    sig = qdef.qsignature[0]
    assert sig.name == "reg"
    # It's a casting register: t1 -> t2
    assert isinstance(sig.dtype, tuple)
    assert len(sig.dtype) == 2
