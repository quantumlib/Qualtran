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

import attrs
import numpy as np
import pytest
import sympy

from qualtran import CtrlSpec, QBit, Side
from qualtran.l1._to_cobject_node import object_to_object_node, to_cobject_node
from qualtran.l1.nodes import CObjectNode, LiteralNode, TupleNode


@attrs.frozen
class SimpleClass:
    x: int
    y: str


@attrs.frozen
class NestedClass:
    s: SimpleClass


def test_literals():
    v1 = to_cobject_node(1)
    assert isinstance(v1, LiteralNode)
    assert v1.value == 1

    v1_5 = to_cobject_node(1.5)
    assert isinstance(v1_5, LiteralNode)
    assert v1_5.value == 1.5

    vfoo = to_cobject_node("foo")
    assert isinstance(vfoo, LiteralNode)
    assert vfoo.value == "foo"


def test_bool_none():
    node = to_cobject_node(True)
    assert isinstance(node, CObjectNode)
    assert node.name == 'True'
    assert not node.cargs

    node = to_cobject_node(False)
    assert isinstance(node, CObjectNode)
    assert node.name == 'False'
    assert not node.cargs

    node = to_cobject_node(None)
    assert isinstance(node, CObjectNode)
    assert node.name == 'None'
    assert not node.cargs


def test_tuple():
    t = (1, "a")
    node = to_cobject_node(t)
    assert isinstance(node, TupleNode)
    assert len(node.items) == 2
    assert isinstance(node.items[0], LiteralNode)
    assert node.items[0].value == 1
    assert isinstance(node.items[1], LiteralNode)
    assert node.items[1].value == "a"


def test_ndarray():
    arr = np.array([1, 2])
    node = to_cobject_node(arr)
    assert isinstance(node, CObjectNode)
    assert node.name == 'NDArr'
    # Check args: shape and data
    args = {arg.key: arg.value for arg in node.cargs}
    assert 'shape' in args
    assert 'data' in args

    # Check shape
    assert isinstance(args['shape'], TupleNode)
    assert isinstance(args['shape'].items[0], LiteralNode)
    assert args['shape'].items[0].value == 2

    # Check data
    assert isinstance(args['data'], TupleNode)
    assert isinstance(args['data'].items[0], LiteralNode)
    assert args['data'].items[0].value == 1
    assert isinstance(args['data'].items[1], LiteralNode)
    assert args['data'].items[1].value == 2


def test_ndarray_too_large():
    arr = np.zeros(101)
    node = to_cobject_node(arr)
    assert isinstance(node, CObjectNode)
    assert node.name == 'Unserializable'
    assert isinstance(node.cargs[0].value, LiteralNode)
    assert node.cargs[0].value.value == 'ndarray_too_large'


def test_attrs_class():
    obj = SimpleClass(x=10, y="hello")
    node = to_cobject_node(obj)
    assert isinstance(node, CObjectNode)
    assert node.name.endswith('SimpleClass')
    assert len(node.cargs) == 2
    assert isinstance(node.cargs[0].value, LiteralNode)
    assert node.cargs[0].value.value == 10
    assert isinstance(node.cargs[1].value, LiteralNode)
    assert node.cargs[1].value.value == "hello"


def test_nested_attrs_class():
    obj = NestedClass(s=SimpleClass(x=1, y="b"))
    node = to_cobject_node(obj)
    assert isinstance(node, CObjectNode)
    assert node.name.endswith('NestedClass')
    assert len(node.cargs) == 1

    inner = node.cargs[0].value
    assert isinstance(inner, CObjectNode)
    assert inner.name.endswith('SimpleClass')


def test_sympy_symbol():
    s = sympy.Symbol('x')
    node = to_cobject_node(s)
    assert isinstance(node, CObjectNode)
    assert node.name == 'Symbol'
    assert isinstance(node.cargs[0].value, LiteralNode)
    assert node.cargs[0].value.value == 'x'


def test_side():
    s = Side.LEFT
    node = to_cobject_node(s)
    assert isinstance(node, CObjectNode)
    assert node.name == 'Side'
    assert isinstance(node.cargs[0].value, LiteralNode)
    assert node.cargs[0].value.value == 'LEFT'


def test_ctrl_spec():
    cs = CtrlSpec(qdtypes=QBit(), cvs=1)
    node = to_cobject_node(cs)
    assert isinstance(node, CObjectNode)
    assert node.name.endswith('CtrlSpec')
    # Should have qdtypes and cvs
    arg_keys = [arg.key for arg in node.cargs]
    assert arg_keys == ['qdtypes', 'cvs']


def test_unserializable_fallback():
    class UnserializableClass:
        pass

    obj = UnserializableClass()
    node = to_cobject_node(obj)

    assert isinstance(node, CObjectNode)
    assert node.name == 'Unserializable'
    assert node.cargs[0].value.value == 'UnserializableClass'  # type: ignore[attr-defined]


def test_to_cobject_node_not_attrs():

    class NonAttrsClass:
        pass

    obj = NonAttrsClass()
    with pytest.raises(TypeError, match="is not an attrs class"):
        object_to_object_node(obj)
