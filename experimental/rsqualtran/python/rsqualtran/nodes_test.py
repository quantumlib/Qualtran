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
from rsqualtran.nodes import (
    AliasAssignmentNode,
    CArgNode,
    CObjectNode,
    CValueNode,
    L1ASTNode,
    L1Module,
    LiteralNode,
    LValueNode,
    QArgNode,
    QArgValueNode,
    QCallNode,
    QCastNode,
    QDefExternNode,
    QDefImplNode,
    QDefNode,
    QDTypeNode,
    QReturnNode,
    QSignatureEntry,
    QStructEntry,
    QStructNode,
    StatementNode,
    TupleNode,
)


def test_literal_node():
    node = LiteralNode(42)
    assert isinstance(node, LiteralNode)
    assert isinstance(node, CValueNode)
    assert isinstance(node, L1ASTNode)
    assert node.value == 42
    assert node.canonical_str() == "42"

    node_str = LiteralNode("hello")
    assert node_str.canonical_str() == "'hello'"

    with pytest.raises(TypeError):
        LiteralNode([1, 2, 3])


def test_tuple_node():
    lit1 = LiteralNode(1)
    lit2 = LiteralNode(2)
    node = TupleNode([lit1, lit2])
    assert isinstance(node, TupleNode)
    assert isinstance(node, CValueNode)
    assert node.canonical_str() == "(1, 2)"

    node_single = TupleNode([lit1])
    assert node_single.canonical_str() == "(1,)"

    with pytest.raises(TypeError):
        TupleNode([1, 2])  # Not CValueNode


def test_carg_node():
    lit = LiteralNode(10)
    node = CArgNode(key="my_key", value=lit)
    assert isinstance(node, CArgNode)
    assert isinstance(node, L1ASTNode)
    assert node.key == "my_key"
    assert node.canonical_str() == "my_key=10"

    node_no_key = CArgNode(key=None, value=lit)
    assert node_no_key.key is None
    assert node_no_key.canonical_str() == "10"

    with pytest.raises(TypeError):
        CArgNode(key=123, value=lit)

    with pytest.raises(TypeError):
        CArgNode(key="k", value=10)


def test_cobject_node():
    carg = CArgNode(key="x", value=LiteralNode(5))
    node = CObjectNode(name="MyObj", cargs=[carg])
    assert isinstance(node, CObjectNode)
    assert isinstance(node, CValueNode)
    assert node.name == "MyObj"
    assert node.canonical_str() == "MyObj(x=5)"

    node_empty = CObjectNode(name="EmptyObj", cargs=[])
    assert node_empty.canonical_str() == "EmptyObj"

    with pytest.raises(TypeError):
        CObjectNode(name=123, cargs=[])

    with pytest.raises(TypeError):
        CObjectNode(name="Obj", cargs=[LiteralNode(1)])


def test_qdtype_node():
    cobj = CObjectNode("QBit", [])
    node = QDTypeNode(dtype=cobj, shape=[2, 2])
    assert isinstance(node, QDTypeNode)
    assert isinstance(node, L1ASTNode)
    assert node.shape == [2, 2]

    node_no_shape = QDTypeNode(dtype=cobj, shape=None)
    assert node_no_shape.shape is None

    with pytest.raises(TypeError):
        QDTypeNode(dtype="QBit", shape=[2])

    with pytest.raises(TypeError):
        QDTypeNode(dtype=cobj, shape=["a"])


def test_qsignature_entry():
    cobj = CObjectNode("QBit", [])
    qdtype = QDTypeNode(dtype=cobj, shape=None)
    node = QSignatureEntry(name="q", dtype=qdtype, annotation=LiteralNode("ann"))
    assert isinstance(node, QSignatureEntry)
    assert isinstance(node, L1ASTNode)
    assert node.name == "q"

    node_tuple = QSignatureEntry(name="q_cast", dtype=(qdtype, None))
    assert isinstance(node_tuple, QSignatureEntry)

    with pytest.raises(TypeError):
        QSignatureEntry(name=123, dtype=qdtype)

    with pytest.raises(TypeError):
        QSignatureEntry(name="q", dtype="invalid")

    with pytest.raises(ValueError):
        QSignatureEntry(name="q", dtype=(qdtype,))


def test_qstruct_entry():
    cobj = CObjectNode("QBit", [])
    qdtype = QDTypeNode(dtype=cobj, shape=None)
    node = QStructEntry(name="field1", dtype=qdtype)
    assert isinstance(node, QStructEntry)
    assert isinstance(node, L1ASTNode)
    assert node.name == "field1"
    assert node.dtype == qdtype


def test_lvalue_node():
    node = LValueNode(name="x", annotation=LiteralNode("ann"))
    assert isinstance(node, LValueNode)
    assert isinstance(node, L1ASTNode)
    assert str(node) == "x @ 'ann'"

    node_no_ann = LValueNode(name="y", annotation=None)
    assert str(node_no_ann) == "y"


def test_statement_nodes():
    alias = AliasAssignmentNode(alias="a", bloq_key="b")
    assert isinstance(alias, AliasAssignmentNode)
    assert isinstance(alias, StatementNode)
    assert isinstance(alias, L1ASTNode)

    with pytest.raises(TypeError):
        AliasAssignmentNode(alias=123, bloq_key="b")

    qav = QArgValueNode(name="q", idx=[0, 1])
    assert isinstance(qav, QArgValueNode)
    assert qav.name == "q"
    assert qav.idx == [0, 1]

    with pytest.raises(TypeError):
        QArgValueNode(name="q", idx=["a"])

    qarg = QArgNode(key="ctrl", value=qav, annotation=LiteralNode("ann"))
    assert isinstance(qarg, QArgNode)
    assert qarg.key == "ctrl"

    with pytest.raises(TypeError):
        QArgNode(key=123, value=qav)

    with pytest.raises(TypeError):
        QArgNode(key="ctrl", value="invalid")

    qcall = QCallNode(
        bloq_key="XGate", lvalues=[LValueNode("ret")], qargs=[qarg], annotation=LiteralNode("ann")
    )
    assert isinstance(qcall, QCallNode)
    assert isinstance(qcall, StatementNode)
    assert qcall.bloq_key == "XGate"
    assert qcall.lvalues == ["ret"]

    with pytest.raises(TypeError):
        QCallNode(bloq_key=123, lvalues=["ret"], qargs=[qarg])

    with pytest.raises(TypeError):
        QCallNode(bloq_key="XGate", lvalues=[123], qargs=[qarg])

    with pytest.raises(TypeError):
        QCallNode(bloq_key="XGate", lvalues=["ret"], qargs=["invalid"])

    qret = QReturnNode(ret_mapping=[qarg])
    assert isinstance(qret, QReturnNode)
    assert isinstance(qret, StatementNode)

    with pytest.raises(TypeError):
        QReturnNode(ret_mapping=["invalid"])


def test_qdef_nodes():
    cobj = CObjectNode("QBit", [])
    qdtype = QDTypeNode(dtype=cobj, shape=None)
    qsig = QSignatureEntry(name="q", dtype=qdtype)
    stmt = AliasAssignmentNode(alias="a", bloq_key="b")
    cobj_from = CObjectNode("XGate", [])

    qimpl = QDefImplNode(bloq_key="MyBloq", qsignature=[qsig], body=[stmt], cobject_from=cobj_from)
    assert isinstance(qimpl, QDefImplNode)
    assert isinstance(qimpl, QDefNode)
    assert isinstance(qimpl, L1ASTNode)

    with pytest.raises(TypeError):
        QDefImplNode(bloq_key=123, qsignature=[qsig], body=[stmt], cobject_from=cobj_from)

    with pytest.raises(TypeError):
        QDefImplNode(bloq_key="MyBloq", qsignature=["invalid"], body=[stmt], cobject_from=cobj_from)

    with pytest.raises(TypeError):
        QDefImplNode(bloq_key="MyBloq", qsignature=[qsig], body=["invalid"], cobject_from=cobj_from)

    qextern = QDefExternNode(bloq_key="ExternBloq", qsignature=[qsig], cobject_from=cobj_from)
    assert isinstance(qextern, QDefExternNode)
    assert isinstance(qextern, QDefNode)

    with pytest.raises(ValueError):
        QDefExternNode(bloq_key="ExternBloq", qsignature=[qsig], cobject_from=None)

    qcast = QCastNode(bloq_key="CastBloq", qsignature=[qsig])
    assert isinstance(qcast, QCastNode)
    assert isinstance(qcast, QDefNode)
    assert qcast.cobject_from is None

    with pytest.raises(TypeError):
        QCastNode(bloq_key=123, qsignature=[qsig])


def test_qstruct_node():
    cobj = CObjectNode("QBit", [])
    qdtype = QDTypeNode(dtype=cobj, shape=None)
    entry = QStructEntry(name="field1", dtype=qdtype)
    node = QStructNode(symbol_id="MyStruct", qfields=[entry], cobject_from=None)
    assert isinstance(node, QStructNode)
    assert isinstance(node, L1ASTNode)
    assert node.symbol_id == "MyStruct"


def test_l1_module():
    cobj = CObjectNode("QBit", [])
    qdtype = QDTypeNode(dtype=cobj, shape=None)
    qsig = QSignatureEntry(name="q", dtype=qdtype)
    qcast = QCastNode(bloq_key="CastBloq", qsignature=[qsig])

    entry = QStructEntry(name="field1", dtype=qdtype)
    qstruct = QStructNode(symbol_id="MyStruct", qfields=[entry], cobject_from=None)

    mod = L1Module(qdefs=[qcast], qstructs=[qstruct])
    assert isinstance(mod, L1Module)
    assert isinstance(mod, L1ASTNode)

    with pytest.raises(TypeError):
        L1Module(qdefs=["invalid"])

    with pytest.raises(TypeError):
        L1Module(qdefs=[qcast], qstructs=["invalid"])
