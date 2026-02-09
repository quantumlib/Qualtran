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
from typing import Any, List, Optional, Sequence

import attrs
import numpy as np
import sympy

from qualtran import CtrlSpec, Side

from .nodes import CArgNode, CObjectNode, CValueNode, LiteralNode, TupleNode


def _get_pkg(cls) -> str:
    """Helper to get the package name of a class."""
    if hasattr(cls, '_pkg_'):
        return cls._pkg_()
    return '.'.join(cls.__module__.split('.')[:-1])


def to_cobject_node(o: object, fieldnames: Optional[Sequence[str]] = None) -> CObjectNode:
    """Convert an object to a CObjectNode.

    This function inspects the fields of the given object
    and recursively converts its values to L1 CValueNode AST nodes.

    The `CObjectNode` can be finally serialized to an objectstring by calling its `canonical_str()`
    method.

    Args:
        o: The object to convert.
        fieldnames: The list of field names to include in the resulting AST node.
            If provided, all fields are saved as 'keyword args'.
            If not provided, the object must be an `attrs` class and all fields defined in
            the attrs class are used, and they are saved either positionally or as keyword
            args according to their `kw_only` attribute.

    Returns:
        A CObjectNode representing the object.

    Raises:
        TypeError: If `o` is not an instance of an attrs class and `fieldnames` is not provided.
    """
    kwonlys: Sequence[bool]
    if fieldnames is not None:
        kwonlys = [True] * len(fieldnames)
    else:
        if not attrs.has(o.__class__):
            raise TypeError(f"{o!r} is not an attrs class")
        fieldnames = tuple(a.name for a in attrs.fields(o.__class__))
        kwonlys = tuple(a.kw_only for a in attrs.fields(o.__class__))

    pos = True  # State machine: whether positional arguments are permitted.
    args: List[CArgNode] = []
    kwargs: List[CArgNode] = []
    for fieldname, kwonly in zip(fieldnames, kwonlys):
        v = getattr(o, fieldname)
        v = any_to_cvalue(v)
        if kwonly:
            pos = False

        if pos:
            args.append(CArgNode(None, v))
        else:
            kwargs.append(CArgNode(fieldname, v))

    pkg = _get_pkg(o.__class__)
    name = o.__class__.__name__
    return CObjectNode(name=f'{pkg}.{name}', cargs=args + kwargs)


def unserializable_object(name: str) -> CObjectNode:
    """Create a CObjectNode representing an unserializable object.

    Args:
        name: The name or description of the unserializable object.

    Returns:
        A CObjectNode with name 'Unserializable' and the given name as an argument.
    """
    return CObjectNode(name='Unserializable', cargs=[CArgNode(None, LiteralNode(name))])


def tuple_to_tupleval(t: tuple) -> TupleNode:
    """Convert a Python tuple to a TupleNode.

    Recursively converts each element of the tuple.

    Args:
        t: The tuple to convert.

    Returns:
        A TupleNode containing the converted elements.
    """
    vals = []
    for v in t:
        v = any_to_cvalue(v)
        vals.append(v)
    return TupleNode(items=vals)


def ndarray_to_ndarr_objectval(a: np.ndarray, max_size: int = 100) -> CObjectNode:
    """Convert a numpy array to a CObjectNode representing an NDArr.

    If the array is too large (> `max_size` elements), it returns an unserializable object node.

    Args:
        a: The numpy array to convert.
        max_size: The maximum number of elements before we give up on serialization.

    Returns:
        A CObjectNode "NDArry" node representing the array (or an unserializable object).
    """
    if a.size > max_size:
        return unserializable_object('ndarray_too_large')

    shape = tuple_to_tupleval(a.shape)
    data = any_to_cvalue(tuple(a.reshape(-1).tolist()))
    return CObjectNode(name='NDArr', cargs=[CArgNode('shape', shape), CArgNode('data', data)])


def any_to_cvalue(x: Any) -> CValueNode:
    """Convert any supported Python value to a CValueNode.

    Include support for `None`, `True`, `False`, integers, strings, floats, tuples, and
    attrs classes.

    Includes custom support for `np.ndarray`, `qlt.CtrlSpec`, `sympy.Symbol`, and `qlt.Side`.

    If an unsupported Python value is encountered, it will return a `CObjectNode` with name
    "Unserializable".

    Args:
        x: The value to convert.

    Returns:
        A CValueNode representing the value.
    """
    if x is None:
        return CObjectNode(name='None', cargs=[])

    if isinstance(x, bool):
        if x:
            return CObjectNode(name='True', cargs=[])
        else:
            return CObjectNode(name='False', cargs=[])

    if isinstance(x, (str, int, float)):
        return LiteralNode(value=x)

    if isinstance(x, tuple):
        return tuple_to_tupleval(x)

    if isinstance(x, np.ndarray):
        return ndarray_to_ndarr_objectval(x)

    if isinstance(x, CtrlSpec):
        return to_cobject_node(x, fieldnames=['qdtypes', 'cvs'])

    if isinstance(x, sympy.Symbol):
        assert isinstance(x.name, str)
        return CObjectNode(name='Symbol', cargs=[CArgNode(None, LiteralNode(x.name))])

    if isinstance(x, Side):
        assert isinstance(x.name, str)
        return CObjectNode(name='Side', cargs=[CArgNode(None, LiteralNode(x.name))])

    if attrs.has(x):
        return to_cobject_node(x)

    return unserializable_object(x.__class__.__name__)
