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

from qualtran.l1._ast_to_code import L1ASTPrinter
from qualtran.l1._parse import parse_module

L1_TEST_CASES = [
    # Empty Module
    "",
    # Extern QDef
    "extern qdef MyBloq\nfrom qualtran.bloqs.MyBloq()\n[    q: t,\n]",
    # Simple QDef Implementation
    """qdef OtherBloq
[
    q: t -> |,
] {
    | = qualtran.bloqs.GlobalPhase()[]
    return [q=q]
}
""",
    # Complex module with multiple QDefs, aliases, array shapes, and nested structures
    """extern qdef ComplexOp
from qualtran.bloqs.ComplexOp(a=1, b='foo')
[    ctrl: t[2, 3],
    trg: t1 -> t2,
    anc: | -> t,
]

qdef MainRoutine
[
    q1: t,
    q2: t[5],
] {
    a = qualtran.bloqs.SomeClass()
    q1, q2 = ComplexOp()[ctrl=q2, trg=q1[0], anc=[[a, b], c]]
    return [q1=q1, q2=q2]
}
""",
]


@pytest.mark.parametrize("original_code", L1_TEST_CASES)
def test_pretty_print_roundtrip(original_code):
    # 1. Parse original source
    original_ast = parse_module(original_code)

    # 2. Pretty print AST back to source string
    pretty_printed_code = L1ASTPrinter().visit(original_ast)

    # 3. Parse pretty printed string
    re_parsed_ast = parse_module(pretty_printed_code)

    # 4. Compare ASTs
    # Use standard assert for equality on the parsed outputs
    assert original_ast == re_parsed_ast


# ---------------------------------------------------------------------------
# Direct-node rendering (annotations, empty tuples, shape validation)
# ---------------------------------------------------------------------------


def _annotation():
    from qualtran.l1.nodes import CObjectNode

    return CObjectNode(name='circle', cargs=[])


def test_lvalue_annotation_rendered():
    from qualtran.l1.nodes import LValueNode

    node = LValueNode(name='q', annotation=_annotation())
    assert L1ASTPrinter().visit(node) == 'q @ circle'


def test_lvalue_without_annotation():
    from qualtran.l1.nodes import LValueNode

    assert L1ASTPrinter().visit(LValueNode(name='q')) == 'q'


def test_qarg_annotation_rendered():
    from qualtran.l1.nodes import QArgNode, QArgValueNode

    node = QArgNode(key='x', value=QArgValueNode(name='q', idx=()), annotation=_annotation())
    assert L1ASTPrinter().visit(node) == 'x=q @ circle'


def test_qcall_annotation_rendered():
    from qualtran.l1.nodes import LValueNode, QArgNode, QArgValueNode, QCallNode

    node = QCallNode(
        bloq_key='B',
        lvalues=[LValueNode(name='q')],
        qargs=[QArgNode(key='x', value=QArgValueNode(name='q', idx=()))],
        annotation=_annotation(),
    )
    _rets, mid, _qargs = L1ASTPrinter().visit(node)
    assert '@ circle' in mid


def test_empty_tuple_node():
    from qualtran.l1.nodes import TupleNode

    assert L1ASTPrinter().visit(TupleNode(items=[])) == '()'


def test_qdtype_node_invalid_shape_raises():
    from qualtran.l1.nodes import CObjectNode, QDTypeNode

    node = QDTypeNode(dtype=CObjectNode(name='QBit', cargs=[]), shape=['not_an_int'])  # type: ignore[list-item]
    with pytest.raises(ValueError, match='Invalid shape'):
        L1ASTPrinter().visit(node)
