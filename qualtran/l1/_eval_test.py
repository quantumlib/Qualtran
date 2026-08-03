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

from qualtran.l1 import eval_module, parse_module, to_cobject_node
from qualtran.l1._eval import (
    _CVALUE_EVALUATORS,
    _eval_imported,
    _PlaceholderBloq,
    _resolve_cobject_dtype,
    BloqError,
    eval_bloq_maybe_aliased,
    eval_carg_nodes,
    eval_cvalue_node,
    eval_ndarr_node,
    eval_qcast_node,
    eval_qdef_extern_node,
    eval_qdef_impl_node,
    eval_qsignature,
    eval_side_node,
    eval_symbol_node,
    eval_unserializable,
    UnevaluatedCValue,
)
from qualtran.l1.nodes import (
    CArgNode,
    CObjectNode,
    LiteralNode,
    QCastNode,
    QDefExternNode,
    QDefImplNode,
    QDTypeNode,
    QSignatureEntry,
    TupleNode,
)


def test_safe_eval_prevents_arbitrary_code():
    # Try to execute os.system('echo hacked')
    # serialized as CObjectNode(name='os.system', cargs=[CArgNode(None, LiteralNode('echo hacked'))])

    node = CObjectNode(name='os.system', cargs=[CArgNode(None, LiteralNode('echo hacked'))])

    # With safe=True, it should return UnevaluatedCValue
    result = eval_cvalue_node(node, safe=True)
    assert isinstance(result, UnevaluatedCValue)
    assert result.name == 'os.system'


def test_safe_eval_allows_whitelisted():
    # `qualtran.Register` is the canonical, allow-listed spelling.
    # We need to construct a node that evaluates to QBit() first.
    from qualtran import Register

    qbit_node = CObjectNode(name='QBit', cargs=[])

    node = CObjectNode(
        name='qualtran.Register',
        cargs=[CArgNode('name', LiteralNode('q')), CArgNode('dtype', qbit_node)],
    )

    result = eval_cvalue_node(node, safe=True)

    assert isinstance(result, Register)
    assert result.name == 'q'


def test_safe_eval_ctrl_spec_canonical():
    # `CtrlSpec` is serialized (and loaded) under its canonical
    # `qualtran.`-qualified name.
    from qualtran import CtrlSpec

    node = CObjectNode(
        name='qualtran.CtrlSpec',
        cargs=[
            CArgNode('qdtypes', CObjectNode(name='QBit', cargs=[])),
            CArgNode('cvs', LiteralNode(1)),
        ],
    )
    result = eval_cvalue_node(node, safe=True)
    assert isinstance(result, CtrlSpec)


def test_safe_eval_adjoint_and_controlled():
    # `Adjoint` and `Controlled` are meta-bloqs allow-listed for safe loading.
    # Their inner bloq argument is resolved recursively via the manifest, so we
    # serialize real instances and confirm they reload under safe=True.
    from qualtran import Adjoint, Controlled, CtrlSpec
    from qualtran.bloqs.basic_gates import TGate, XGate

    meta_bloqs = [(Adjoint(TGate()), Adjoint), (Controlled(XGate(), CtrlSpec()), Controlled)]
    for bloq, cls in meta_bloqs:
        node = to_cobject_node(bloq)
        result = eval_cvalue_node(node, safe=True)
        assert isinstance(result, cls)


@pytest.mark.parametrize('name', ['CtrlSpec', 'Register', 'Adjoint', 'Controlled'])
def test_safe_eval_rejects_bare_infra_names(name):
    # The bare spelling is not canonical and must not resolve under safe=True;
    # it degrades to an UnevaluatedCValue rather than being imported.
    node = CObjectNode(name=name, cargs=[])
    result = eval_cvalue_node(node, safe=True)
    assert isinstance(result, UnevaluatedCValue)


def test_safe_eval_symbol_enforces_string():
    # Symbol with non-string arg should fail in safe mode
    node = CObjectNode(name='Symbol', cargs=[CArgNode(None, LiteralNode(123))])

    with pytest.raises(TypeError, match="Symbol nodes must take one string argument"):
        eval_cvalue_node(node, safe=True)


def test_safe_eval_propagates_safe_flag():
    # Nested object that is not whitelisted
    inner_node = CObjectNode(name='os.system', cargs=[CArgNode(None, LiteralNode('echo hacked'))])

    # Tuple containing the unsafe node
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
    from qualtran import Side

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
    # Verify that safe=False propagates through eval_module into impl qdefs.

    # Regression test: eval_bloq_maybe_aliased was not forwarding the `safe`
    # flag when calling eval_qdef_impl_node for QDefImplNode, causing the
    # `from` clause to silently default to safe=True and return
    # UnevaluatedCValue for non-manifest dotted names.

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
    from qualtran import CompositeBloq

    assert isinstance(safe_bloq, CompositeBloq)
    assert safe_bloq.decomposed_from is None

    # With safe=False, the `from` clause should fully resolve to an
    # OrderedDict instance (not a Bloq, but not UnevaluatedCValue either).
    # decomposed_from will still be None because OrderedDict isn't a Bloq,
    # but we can verify the resolution by calling eval_cvalue_node directly
    # on the impl qdef's cobject_from node.
    impl_qdef = [q for q in mod.qdefs if q.bloq_key == 'MyBloq'][0]
    assert impl_qdef.cobject_from is not None
    result = eval_cvalue_node(impl_qdef.cobject_from, safe=False)
    import collections

    assert isinstance(
        result, collections.OrderedDict
    ), f"Expected OrderedDict, got {type(result).__name__}: {result!r}"

    # Verify that safe=True returns UnevaluatedCValue for the same node.
    result_safe_direct = eval_cvalue_node(
        impl_qdef.cobject_from, safe=True
    )  # cobject_from asserted not None above
    assert isinstance(result_safe_direct, UnevaluatedCValue)


def test_eval_qcast_node():
    from qualtran.bloqs.bookkeeping.qcast import QCast

    # Build a QCastNode for Split(QUInt(4)): reg: QUInt(4) -> QBit[4]
    quint4 = QDTypeNode(
        dtype=CObjectNode(name='QUInt', cargs=[CArgNode(None, LiteralNode(4))]), shape=None
    )
    qbit_arr = QDTypeNode(dtype=CObjectNode(name='QBit', cargs=[]), shape=[4])
    sig_entry = QSignatureEntry(name='reg', dtype=(quint4, qbit_arr))
    qcast_node = QCastNode(bloq_key='Split(QUInt(4))', qsignature=[sig_entry])

    result = eval_qcast_node(qcast_node, safe=True)
    assert isinstance(result, QCast)
    assert len(result.signature) == 2  # LEFT reg and RIGHT reg


def test_qcast_roundtrip():
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


# ---------------------------------------------------------------------------
# _resolve_cobject_dtype
# ---------------------------------------------------------------------------


def test_resolve_cobject_dtype_unknown_name():
    with pytest.raises(ValueError, match='Unknown data type'):
        _resolve_cobject_dtype(CObjectNode(name='NotADtype', cargs=[]), safe=True)


def test_resolve_cobject_dtype_dotted_not_on_manifest_safe():
    with pytest.raises(ImportError, match='not in the bloq manifest'):
        _resolve_cobject_dtype(CObjectNode(name='some.fake.Dtype', cargs=[]), safe=True)


def test_resolve_cobject_dtype_dotted_import_unsafe():
    from qualtran import QUInt

    # A dotted name is importable directly when safe=False.
    result = _resolve_cobject_dtype(
        CObjectNode(name='qualtran.QUInt', cargs=[CArgNode(None, LiteralNode(4))]), safe=False
    )
    assert result == QUInt(4)


# ---------------------------------------------------------------------------
# eval_qsignature
# ---------------------------------------------------------------------------


def test_eval_qsignature_bad_tuple_length():
    entry = QSignatureEntry(name='x', dtype=(None,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match='Expected a 2-tuple'):
        eval_qsignature([entry], safe=True)


def test_eval_qsignature_both_sides_none():
    entry = QSignatureEntry(name='x', dtype=(None, None))
    with pytest.raises(ValueError, match='Both LEFT and RIGHT'):
        eval_qsignature([entry], safe=True)


# ---------------------------------------------------------------------------
# eval_qdef_extern_node
# ---------------------------------------------------------------------------


def test_eval_qdef_extern_node_missing_from():
    qdef = QDefExternNode(bloq_key='X', qsignature=[], cobject_from=None)
    with pytest.raises(ValueError, match='`from` clause is required'):
        eval_qdef_extern_node(qdef, safe=True)


def test_eval_qdef_extern_node_from_not_a_bloq_returns_placeholder():
    # The `from` clause resolves to a QBit dtype, which is not a Bloq.
    qdef = QDefExternNode(
        bloq_key='X', qsignature=[], cobject_from=CObjectNode(name='QBit', cargs=[])
    )
    with pytest.warns(UserWarning, match='instead of a Bloq object'):
        result = eval_qdef_extern_node(qdef, safe=True)
    assert isinstance(result, _PlaceholderBloq)


# ---------------------------------------------------------------------------
# eval_bloq_maybe_aliased
# ---------------------------------------------------------------------------


def test_eval_bloq_maybe_aliased_unresolvable():
    with pytest.raises(ValueError, match='Could not resolve'):
        eval_bloq_maybe_aliased('nope', {}, {}, {}, safe=True)


def test_eval_bloq_maybe_aliased_unknown_qdef_type():
    # A value that is neither an extern, qcast, nor impl qdef.
    qdefs = {'x': object()}
    with pytest.raises(TypeError, match='Unknown qdef type'):
        eval_bloq_maybe_aliased('x', qdefs, {}, {}, safe=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# eval_qdef_impl_node
# ---------------------------------------------------------------------------


def test_eval_qdef_impl_node_missing_return():
    qdef = QDefImplNode(bloq_key='x', qsignature=[], body=[], cobject_from=None)
    with pytest.raises(ValueError, match='lacks a `return`'):
        eval_qdef_impl_node(qdef, {}, {}, safe=True)


def test_eval_qdef_impl_node_bad_statement():
    qdef = QDefImplNode(
        bloq_key='x', qsignature=[], body=[object()], cobject_from=None  # type: ignore[list-item]
    )
    with pytest.raises(ValueError, match='Bad stmt'):
        eval_qdef_impl_node(qdef, {}, {}, safe=True)


def test_eval_qdef_extern_node_from_clause_raises():
    # With safe=False, a bad dotted import raises inside eval_cvalue_node,
    # which is caught and downgraded to a placeholder with a warning.
    qdef = QDefExternNode(
        bloq_key='X',
        qsignature=[],
        cobject_from=CObjectNode(name='nonexistent_module_xyz.Foo', cargs=[]),
    )
    with pytest.warns(UserWarning):
        result = eval_qdef_extern_node(qdef, safe=False)
    assert isinstance(result, _PlaceholderBloq)


def test_eval_qcall_return_count_mismatch():
    # XGate returns a single wire, but the call writes two lvalues.
    l1_code = """# Qualtran-L1
# 1.0.0

extern qdef XGate
from qualtran.bloqs.basic_gates.XGate()
[q: QBit()]

qdef Bad
[q: QBit()] {
    a, b = XGate [q=q]
    return [q=a]
}
"""
    mod = parse_module(l1_code)
    with pytest.raises(BloqError, match='return values'):
        eval_module(mod, safe=True)
