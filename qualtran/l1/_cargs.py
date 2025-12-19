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
from typing import Any, List, Optional, Sequence

import attrs
import numpy as np
import sympy

from qualtran import CtrlSpec, Side

from .nodes import CArgNode, CObjectNode, CValueNode, LiteralNode, TupleNode


def _get_pkg(cls):
    if hasattr(cls, '_pkg_'):
        return cls._pkg_()
    return '.'.join(cls.__module__.split('.')[:-1])


def object_to_cobject_node(o: object, fieldnames: Optional[Sequence[str]] = None) -> CObjectNode:
    if fieldnames is not None:
        kwonlys = [True] * len(fieldnames)
    else:
        if not attrs.has(o.__class__):
            raise TypeError(f"{o!r} is not an attrs class")
        fieldnames = (a.name for a in attrs.fields(o.__class__))
        kwonlys = (a.kw_only for a in attrs.fields(o.__class__))

    pos = True
    args: List[CArgNode] = []
    kwargs: List[CArgNode] = []
    for fieldname, kwonly in zip(fieldnames, kwonlys):
        v = getattr(o, fieldname)
        v = any_to_cargvalue(v)
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
    return CObjectNode(name='Unserializable', cargs=[CArgNode(None, LiteralNode(name))])


def tuple_to_tupleval(t: tuple) -> CValueNode:
    vals = []
    for v in t:
        v = any_to_cargvalue(v)
        vals.append(v)
    return TupleNode(items=vals)


def ndarray_to_ndarr_objectval(a: np.ndarray) -> CObjectNode:
    if a.size > 100:
        return unserializable_object('ndarray_too_large')

    shape = tuple_to_tupleval(a.shape)
    data = any_to_cargvalue(tuple(a.reshape(-1).tolist()))
    return CObjectNode(name='NDArr', cargs=[CArgNode('shape', shape), CArgNode('data', data)])


def any_to_cargvalue(x: Any) -> CValueNode:
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
        return object_to_cobject_node(x, fieldnames=['qdtypes', 'cvs'])

    if isinstance(x, sympy.Symbol):
        return CObjectNode(name='Symbol', cargs=[CArgNode(None, LiteralNode(x.name))])

    if isinstance(x, Side):
        return CObjectNode(name='Side', cargs=[CArgNode(None, LiteralNode(x.name))])

    if attrs.has(x):
        return object_to_cobject_node(x)

    return unserializable_object(x.__class__.__name__)
