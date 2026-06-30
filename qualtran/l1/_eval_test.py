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

import numpy as np
import pytest
import sympy

from qualtran import Register, Side
from qualtran.l1._eval import (
    _CVALUE_EVALUATORS,
    _eval_imported,
    eval_carg_nodes,
    eval_cvalue_node,
    eval_ndarr_node,
    eval_side_node,
    eval_symbol_node,
    eval_unserializable,
    UnevaluatedCValue,
)
from qualtran.l1.nodes import CArgNode, CObjectNode, LiteralNode, TupleNode


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


def test_eval_carg_nodes_positional_after_keyword():
    with pytest.raises(ValueError, match="Positional argument after keyword"):
        eval_carg_nodes([CArgNode('key', LiteralNode(1)), CArgNode(None, LiteralNode(2))])


def test_eval_ndarr_node_invalid_args():
    with pytest.raises(TypeError, match="nodes should be keyword only"):
        eval_ndarr_node(
            CObjectNode(name='NDArr', cargs=[CArgNode(None, LiteralNode(1))]), safe=True
        )

    with pytest.raises(TypeError, match="nodes should have elements"):
        eval_ndarr_node(
            CObjectNode(name='NDArr', cargs=[CArgNode('not_shape', LiteralNode(1))]), safe=True
        )


def test_eval_symbol_node_invalid_args():
    with pytest.raises(TypeError, match="nodes should be positional only"):
        eval_symbol_node(
            CObjectNode(name='Symbol', cargs=[CArgNode('k', LiteralNode(1))]), safe=True
        )

    with pytest.raises(TypeError, match="must have one argument"):
        eval_symbol_node(CObjectNode(name='Symbol', cargs=[]), safe=True)


def test_eval_symbol_node_valid():
    res = eval_symbol_node(
        CObjectNode(name='Symbol', cargs=[CArgNode(None, LiteralNode('x'))]), safe=True
    )
    assert isinstance(res, sympy.Symbol)
    assert res.name == 'x'


def test_eval_side_node():

    with pytest.raises(TypeError, match="Side nodes should be positional only"):
        eval_side_node(CObjectNode(name='Side', cargs=[CArgNode('k', LiteralNode(1))]), safe=True)

    with pytest.raises(TypeError, match="Side nodes must have one argument"):
        eval_side_node(CObjectNode(name='Side', cargs=[]), safe=True)

    res = eval_side_node(
        CObjectNode(name='Side', cargs=[CArgNode(None, LiteralNode('RIGHT'))]), safe=True
    )
    assert res == Side.RIGHT


def test_unserializable():
    with pytest.raises(ValueError, match="properly serialized"):
        eval_unserializable(CObjectNode(name='Unserializable', cargs=[]), safe=True)


def test_singleton_with_args():
    with pytest.raises(ValueError, match="should not have any arguments"):
        _CVALUE_EVALUATORS['True'](
            CObjectNode(name='True', cargs=[CArgNode(None, LiteralNode(1))]), safe=True
        )

    # Test valid singleton evaluation
    assert _CVALUE_EVALUATORS['True'](CObjectNode(name='True', cargs=[]), safe=True) is True
    assert _CVALUE_EVALUATORS['False'](CObjectNode(name='False', cargs=[]), safe=True) is False
    assert _CVALUE_EVALUATORS['None'](CObjectNode(name='None', cargs=[]), safe=True) is None


def test_eval_imported():
    import qualtran.bloqs.basic_gates

    # test we can import and instantiate XGate
    res = _eval_imported('qualtran.bloqs.basic_gates.XGate', [], {})
    assert isinstance(res, qualtran.bloqs.basic_gates.XGate)


def test_eval_unsafe():
    # evaluate unknown unsafe class
    node = CObjectNode(
        name='qualtran.bloqs.mcmt.and_bloq.And',
        cargs=[CArgNode('cv1', LiteralNode(1)), CArgNode('cv2', LiteralNode(1))],
    )
    res = eval_cvalue_node(node, safe=False)
    from qualtran.bloqs.mcmt.and_bloq import And

    assert isinstance(res, And)

    # Try one without dot
    node2 = CObjectNode(name='Unknown', cargs=[])
    with pytest.raises(ValueError, match="Unknown CValueNode"):
        eval_cvalue_node(node2, safe=False)

    # Try one with dot but not in allowlist
    node3 = CObjectNode(name='collections.Counter', cargs=[])
    res3 = eval_cvalue_node(node3, safe=False)
    import collections

    assert isinstance(res3, collections.Counter)


def test_eval_ndarr_node_valid():
    node = CObjectNode(
        name='NDArr',
        cargs=[
            CArgNode('shape', TupleNode([LiteralNode(1)])),
            CArgNode('data', TupleNode([LiteralNode(5)])),
        ],
    )
    res = eval_ndarr_node(node, safe=True)
    assert isinstance(res, np.ndarray)
    assert res.shape == (1,)
    assert res[0] == 5


def test_imported_bloq_class():
    node = CObjectNode(name='qualtran.bloqs.basic_gates.XGate', cargs=[])
    res = eval_cvalue_node(node, safe=True)
    from qualtran.bloqs.basic_gates import XGate

    assert isinstance(res, XGate)


def test_eval_cvalue_node_type_error():
    with pytest.raises(TypeError, match="Unknown AST node type"):
        eval_cvalue_node("just a string")  # type: ignore[arg-type]


def test_too_many_args():
    node = CObjectNode(name='Test', cargs=[CArgNode(None, LiteralNode(5))] * 1_001)

    with pytest.raises(ValueError, match=r'Too many.*'):
        _ = eval_cvalue_node(node, safe=True)


def test_too_many_values():
    node = TupleNode([LiteralNode(5)] * 1_001)

    with pytest.raises(ValueError, match=r'Too many.*'):
        _ = eval_cvalue_node(node, safe=True)


def test_safe_false_propagates_to_impl_qdefs():
    """Verify that safe=False propagates through eval_module into impl qdefs.

    Regression test: eval_bloq_maybe_aliased was not forwarding the `safe`
    flag when calling eval_qdef_impl_node for QDefImplNode, causing the
    `from` clause to silently default to safe=True and return
    UnevaluatedCValue for non-manifest dotted names.
    """
    from qualtran.l1 import eval_module, parse_module
    from qualtran.l1._eval import _eval_qdef_impl_node

    # An impl qdef whose `from` clause uses a dotted name NOT in the bloq
    # manifest (collections.OrderedDict is a harmless stdlib class).
    l1_code = """\
# Qualtran-L1
# 1.0.0

extern qdef XGate
from qualtran.bloqs.basic_gates.XGate()
[q: QBit()]

qdef MyBloq
from collections.OrderedDict()
[q: QBit()] {
    q = XGate [q=q]
    return [q=q]
}
"""
    mod = parse_module(l1_code)

    # With safe=True, the `from` clause of the impl qdef should fail to
    # resolve (OrderedDict is not on the manifest), producing an
    # UnevaluatedCValue internally. The bloq still builds but
    # decomposed_from is None.
    result_safe = eval_module(mod, safe=True)
    safe_bloq = result_safe['MyBloq']
    assert safe_bloq.decomposed_from is None

    # With safe=False, the `from` clause should fully resolve to an
    # OrderedDict instance (not a Bloq, but not UnevaluatedCValue either).
    # decomposed_from will still be None because OrderedDict isn't a Bloq,
    # but we can verify the resolution by calling eval_cvalue_node directly
    # on the impl qdef's cobject_from node.
    impl_qdef = [q for q in mod.qdefs if q.bloq_key == 'MyBloq'][0]
    result = eval_cvalue_node(impl_qdef.cobject_from, safe=False)
    import collections

    assert isinstance(
        result, collections.OrderedDict
    ), f"Expected OrderedDict, got {type(result).__name__}: {result!r}"

    # Verify that safe=True returns UnevaluatedCValue for the same node.
    result_safe_direct = eval_cvalue_node(impl_qdef.cobject_from, safe=True)
    assert isinstance(result_safe_direct, UnevaluatedCValue)


def test_eval_qcast_node():
    from qualtran.l1._eval import eval_qcast_node
    from qualtran.l1.nodes import CObjectNode, QCastNode, QDTypeNode, QSignatureEntry

    from qualtran.bloqs.bookkeeping.qcast import QCast

    # Build a QCastNode for Split(QUInt(4)): reg: QUInt(4) -> QBit[4]
    quint4 = QDTypeNode(dtype=CObjectNode(name='QUInt', cargs=[CArgNode(None, LiteralNode(4))]), shape=None)
    qbit_arr = QDTypeNode(dtype=CObjectNode(name='QBit', cargs=[]), shape=[4])
    sig_entry = QSignatureEntry(name='reg', dtype=(quint4, qbit_arr))
    qcast_node = QCastNode(bloq_key='Split(QUInt(4))', qsignature=[sig_entry])

    result = eval_qcast_node(qcast_node, safe=True)
    assert isinstance(result, QCast)
    assert len(result.signature) == 2  # LEFT reg and RIGHT reg


def test_qcast_roundtrip():
    from qualtran.l1 import eval_module, parse_module
    from qualtran.bloqs.bookkeeping.qcast import QCast

    l1_code = """# Qualtran-L1
# 1.0.0

qcast Split(QUInt(4))
[reg: QUInt(4) -> QBit[4]]
"""
    mod = parse_module(l1_code)
    bloqs = eval_module(mod, safe=True)
    assert 'Split(QUInt(4))' in bloqs
    bloq = bloqs['Split(QUInt(4))']
    assert isinstance(bloq, QCast)
