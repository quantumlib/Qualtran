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

from qualtran import CtrlSpec, QDType, Side

from . import nodes as qualtran_l1_nodes
from ._dtypes import get_builtin_qdtypes
from .nodes import CArgNode, CObjectNode, CValueNode, L1Nodes, TupleNode


def _get_pkg(cls) -> str:
    """Helper to get the package name of a class."""
    if hasattr(cls, '_pkg_'):
        return cls._pkg_()
    return '.'.join(cls.__module__.split('.')[:-1])


def object_to_object_node(
    o: object,
    *,
    fieldnames: Optional[Sequence[str]] = None,
    pkg: Optional[str] = None,
    nodes: L1Nodes = qualtran_l1_nodes,
) -> CObjectNode:
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
        pkg: The dot-delimited package string to use in the resultant AST node's name. If
            the provided value is the empty string, no package name will be included. If
            there is no value provided, a package string will be deduced from 1) a _pkg_
            classmethod on the object, if present or 2) the class's `__module__` parts,
            excluding the final name. This matches the Qualtran convention of re-exporting
            bloq classes one level up.
        nodes: The module providing the AST node constructors.

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
        v = to_cobject_node(v, nodes=nodes)
        if kwonly:
            pos = False

        if pos:
            args.append(nodes.CArgNode(None, v))
        else:
            kwargs.append(nodes.CArgNode(fieldname, v))

    pkg = _get_pkg(o.__class__) if pkg is None else str(pkg)
    name = o.__class__.__name__
    qualname = f'{pkg}.{name}' if pkg else name
    return nodes.CObjectNode(name=f'{qualname}', cargs=args + kwargs)


def unserializable_object(name: str, *, nodes: L1Nodes = qualtran_l1_nodes) -> CObjectNode:
    """Create a CObjectNode representing an unserializable object.

    Args:
        name: The name or description of the unserializable object.
        nodes: The module providing the AST node constructors.

    Returns:
        A CObjectNode with name 'Unserializable' and the given name as an argument.
    """
    return nodes.CObjectNode(
        name='Unserializable', cargs=[nodes.CArgNode(None, nodes.LiteralNode(name))]
    )


def tuple_to_tuplenode(t: tuple, *, nodes: L1Nodes = qualtran_l1_nodes) -> TupleNode:
    """Convert a Python tuple to a TupleNode.

    Recursively converts each element of the tuple.

    Args:
        t: The tuple to convert.
        nodes: The module providing the AST node constructors.

    Returns:
        A TupleNode containing the converted elements.
    """
    vals = []
    for v in t:
        v = to_cobject_node(v, nodes=nodes)
        vals.append(v)
    return nodes.TupleNode(items=vals)


def ndarray_to_ndarr_objectnode(
    a: np.ndarray, max_size: int = 100, *, nodes: L1Nodes = qualtran_l1_nodes
) -> CObjectNode:
    """Convert a numpy array to a CObjectNode representing an NDArr.

    If the array is too large (> `max_size` elements), it returns an unserializable object node.

    Args:
        a: The numpy array to convert.
        max_size: The maximum number of elements before we give up on serialization.
        nodes: The module providing the AST node constructors.

    Returns:
        A CObjectNode "NDArry" node representing the array (or an unserializable object).
    """
    if a.size > max_size:
        return unserializable_object('ndarray_too_large', nodes=nodes)

    shape = tuple_to_tuplenode(a.shape, nodes=nodes)
    data = to_cobject_node(tuple(a.reshape(-1).tolist()), nodes=nodes)
    return nodes.CObjectNode(
        name='NDArr', cargs=[nodes.CArgNode('shape', shape), nodes.CArgNode('data', data)]
    )


def to_cobject_node(x: Any, *, nodes: L1Nodes = qualtran_l1_nodes) -> CValueNode:
    """Convert any supported Python value to a CValueNode.

    Include support for `None`, `True`, `False`, integers, strings, floats, tuples, and
    attrs classes.

    Includes custom support for `np.ndarray`, `qlt.CtrlSpec`, `sympy.Symbol`, and `qlt.Side`.

    If an unsupported Python value is encountered, it will return a `CObjectNode` with name
    "Unserializable".

    Args:
        x: The value to convert.
        nodes: The module providing the AST node constructors.

    Returns:
        A CValueNode representing the value.
    """
    if x is None:
        return nodes.CObjectNode(name='None', cargs=[])

    # Normalize numpy scalar types to their native Python equivalents so they
    # serialize as plain literals rather than as `Unserializable`. This is
    # important because e.g. `CtrlSpec` control values are frequently stored as
    # numpy integers (`np.uint8`) rather than Python `int`s.
    if isinstance(x, np.bool_):
        x = bool(x)
    elif isinstance(x, np.integer):
        x = int(x)
    elif isinstance(x, np.floating):
        x = float(x)

    if isinstance(x, bool):
        if x:
            return nodes.CObjectNode(name='True', cargs=[])
        else:
            return nodes.CObjectNode(name='False', cargs=[])

    if isinstance(x, (str, int, float)):
        return nodes.LiteralNode(value=x)

    if isinstance(x, tuple):
        return tuple_to_tuplenode(x, nodes=nodes)

    if isinstance(x, np.ndarray):
        return ndarray_to_ndarr_objectnode(x, nodes=nodes)

    if isinstance(x, CtrlSpec):
        return object_to_object_node(x, fieldnames=['qdtypes', 'cvs'], nodes=nodes)

    if isinstance(x, sympy.Symbol):
        assert isinstance(x.name, str)
        return nodes.CObjectNode(
            name='Symbol', cargs=[nodes.CArgNode(None, nodes.LiteralNode(x.name))]
        )

    if isinstance(x, Side):
        assert isinstance(x.name, str)
        return nodes.CObjectNode(
            name='Side', cargs=[nodes.CArgNode(None, nodes.LiteralNode(x.name))]
        )

    if isinstance(x, QDType) and isinstance(x, get_builtin_qdtypes()):
        return object_to_object_node(x, pkg='', nodes=nodes)

    if attrs.has(x):
        return object_to_object_node(x, nodes=nodes)

    return unserializable_object(x.__class__.__name__, nodes=nodes)


def dump_objectstring(x: Any, *, nodes: L1Nodes = qualtran_l1_nodes) -> str:
    return to_cobject_node(x, nodes=nodes).canonical_str()
