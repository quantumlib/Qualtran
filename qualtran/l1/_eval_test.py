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
