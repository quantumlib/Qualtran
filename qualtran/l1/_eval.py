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
import importlib
import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, TYPE_CHECKING

import attrs
import numpy as np
import sympy

from ..bloqs.manifest import BLOQ_CLASS_NAMES
from ._dtypes import get_builtin_qdtypes
from .nodes import CArgNode, CObjectNode, CValueNode, LiteralNode, TupleNode

if TYPE_CHECKING:
    import qualtran

logger = logging.getLogger(__name__)


@lru_cache
def _get_safe_loadables() -> Dict[str, type[Any]]:
    from qualtran import CtrlSpec, Register

    return {'CtrlSpec': CtrlSpec, 'Register': Register}


def eval_carg_nodes(
    cargs: Sequence[CArgNode], *, safe: bool = True
) -> Tuple[List[Any], Dict[str, Any]]:
    """Evaluate a sequence of `CArgNode`

    Returns:
        args: The evaluated positional arguments
        kwargs: The evaluated keyword arguments
    """
    args: List[Any] = []
    kwargs: Dict[str, Any] = {}
    kwarg_only = False
    for arg in cargs:
        if arg.key is None:
            if kwarg_only:
                raise ValueError(f"Positional argument after keyword argument in {cargs}")
            args.append(eval_cvalue_node(arg.value, safe=safe))
        else:
            kwargs[arg.key] = eval_cvalue_node(arg.value, safe=safe)
            kwarg_only = True

    return args, kwargs


class _EvalProtocol(Protocol):
    def __call__(self, node: CObjectNode, *, safe: bool) -> Any: ...


def eval_ndarr_node(node: CObjectNode, *, safe: bool) -> np.ndarray:
    """Evaluate a CObjectNode where `name` is `"NDArr"` to a `np.ndarray`."""
    args, kwargs = eval_carg_nodes(node.cargs, safe=safe)
    if args:
        raise TypeError(f"NDArr nodes should be keyword only: {node}")
    if not set(kwargs.keys()) == {'shape', 'data'}:
        raise TypeError(f"NDArr nodes should have elements 'shape' and 'data': {node}")

    return np.array(kwargs['data']).reshape(kwargs['shape'])


def eval_symbol_node(node: CObjectNode, *, safe: bool) -> sympy.Symbol:
    """Evaluate a CObjectNode where `name` is `"Symbol"` to a `sympy.Symbol`"""
    args, kwargs = eval_carg_nodes(node.cargs, safe=safe)
    if kwargs:
        raise TypeError(f"Symbol nodes should be positional only: {node}")
    if len(args) != 1:
        raise TypeError(f"Symbol nodes must have one argument, not {args}")
    arg = args[0]
    if safe and type(arg) != str:
        raise TypeError(f"Symbol nodes must take one string argument, not {arg}")
    return sympy.Symbol(args[0])


def eval_side_node(node: CObjectNode, *, safe: bool) -> 'qualtran.Side':
    """Evaluate a CObjectNode where `name` is `"Side"` to a `qualtran.Side`"""
    from qualtran import Side

    args, kwargs = eval_carg_nodes(node.cargs, safe=safe)
    if kwargs:
        raise TypeError(f"Side nodes should be positional only: {node}")
    if len(args) != 1:
        raise TypeError(f"Side nodes must have one argument, not {args}")
    return Side[args[0]]


def eval_unserializable(node: CObjectNode, *, safe: bool):
    """Fail to evaluate a CObjectNode where `name` is `"Unserializable"`"""
    raise ValueError(f"Tried to evaluate an AST node that was not properly serialized: {node}")


def _get_singleton_evaluator(o: object) -> _EvalProtocol:
    """Helper to return an evaluator function for singletons."""

    def _eval_node(node: CObjectNode, *, safe: bool) -> object:
        if node.cargs:
            raise ValueError(f"{node} should not have any arguments")
        return o

    return _eval_node


_CVALUE_EVALUATORS: Dict[str, _EvalProtocol] = {
    'True': _get_singleton_evaluator(True),
    'False': _get_singleton_evaluator(False),
    'None': _get_singleton_evaluator(None),
    'Unserializable': eval_unserializable,
    'NDArr': eval_ndarr_node,
    'Symbol': eval_symbol_node,
    'Side': eval_side_node,
}


@attrs.frozen
class UnevaluatedCValue:
    name: str
    cargs: Sequence[Tuple[Optional[str], Any]] = attrs.field(
        converter=tuple[Tuple[Optional[str], Any]]
    )


def _eval_imported(name: str, args: Sequence[Any], kwargs: Dict[str, Any]):
    name_parts = name.split('.')
    package = '.'.join(name_parts[:-1])
    logger.debug("importing %s", package)
    module = importlib.import_module(package)
    cls = getattr(module, name_parts[-1])
    return cls(*args, **kwargs)


def eval_cvalue_node(node: CValueNode, *, safe: bool = True) -> Any:
    # Literals
    if isinstance(node, LiteralNode):
        return node.value

    # Recurse on tuples
    if isinstance(node, TupleNode):
        return tuple(eval_cvalue_node(n, safe=safe) for n in node.items)

    # Objects
    safe_context = get_builtin_qdtypes() | _get_safe_loadables()
    if isinstance(node, CObjectNode):
        # Known speciality types
        if node.name in _CVALUE_EVALUATORS:
            eval_func = _CVALUE_EVALUATORS[node.name]
            return eval_func(node, safe=safe)

        # Known generic constructors
        args, kwargs = eval_carg_nodes(node.cargs, safe=safe)
        if node.name in safe_context:
            cls = safe_context[node.name]
            return cls(*args, **kwargs)

        # Allowlisted importable bloq classes
        if '.' in node.name and node.name in BLOQ_CLASS_NAMES:
            return _eval_imported(node.name, args, kwargs)

        # Unknown (safe mode): return unevaluated
        if safe:
            uneval_cargs: list[tuple[Optional[str], Any]] = [(None, arg) for arg in args]
            uneval_cargs.extend((k, v) for k, v in kwargs.items())
            return UnevaluatedCValue(name=node.name, cargs=uneval_cargs)

        # Unknown, unsafe: use `importlib` to load the class.
        if '.' not in node.name:
            raise ValueError(f"Unknown CValueNode {node}.")
        return _eval_imported(node.name, args, kwargs)

    raise TypeError(f"Unknown AST node type: {type(node)}")
