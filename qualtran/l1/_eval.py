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
from typing import (
    Any,
    cast,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    TypeAlias,
    Union,
)

import attrs
import numpy as np
import sympy

from qualtran import (
    Bloq,
    BloqBuilder,
    BloqError,
    CompositeBloq,
    QCDType,
    Register,
    Side,
    Signature,
    SoquetT,
)
from qualtran.bloqs.manifest import BLOQ_CLASS_NAMES

from ._dtypes import get_builtin_qdtype_mapping
from .nodes import (
    AliasAssignmentNode,
    CArgNode,
    CObjectNode,
    CValueNode,
    L1Module,
    LiteralNode,
    NestedQArgValue,
    QArgNode,
    QArgValueNode,
    QCallNode,
    QDefExternNode,
    QDefImplNode,
    QDefNode,
    QDTypeNode,
    QReturnNode,
    QSignatureEntry,
    StatementNode,
    TupleNode,
)

if TYPE_CHECKING:
    import qualtran

logger = logging.getLogger(__name__)

BloqKey: TypeAlias = str


@lru_cache
def _get_custom_dtypes() -> Dict[str, type[QCDType]]:
    # TODO: Custom data types.
    dtypes: Dict[str, type[QCDType]] = {f'{k._pkg_()}.{k.__name__}': k for k in []}  # type: ignore
    return dtypes


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
    if len(cargs) > 1_000 and safe:
        raise ValueError("Too many arguments for safe=True loading.")
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
    if safe and type(arg) is not str:
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
        if len(node.items) > 1_000 and safe:
            raise ValueError("Too many elements for safe=True tuple.")
        return tuple(eval_cvalue_node(n, safe=safe) for n in node.items)

    # Objects
    safe_context = get_builtin_qdtype_mapping() | _get_safe_loadables()
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


def eval_qdtype_node(dt: QDTypeNode) -> Tuple['qualtran.QCDType', Sequence[int]]:
    context = get_builtin_qdtype_mapping() | _get_custom_dtypes()
    try:
        dt_cls = context[dt.dtype.name]
    except KeyError as e:
        raise ValueError(f"Unknown data type {dt.dtype.name}") from e
    args, kwargs = eval_carg_nodes(dt.dtype.cargs)
    qdt = dt_cls(*args, **kwargs)

    if dt.shape is not None:
        qshape = tuple(v for v in dt.shape)
    else:
        qshape = ()
    return qdt, qshape


def eval_qsignature(entries: Sequence[QSignatureEntry]) -> 'qualtran.Signature':
    registers = []
    for entry in entries:
        # One dtype: THRU register.
        if isinstance(entry.dtype, QDTypeNode):
            dtype, shape = eval_qdtype_node(entry.dtype)
            registers.append(
                Register(name=entry.name, dtype=dtype, side=Side.THRU, shape=tuple(shape))
            )
            continue

        # Two dtypes: LEFT and/or RIGHT registers
        assert isinstance(entry.dtype, tuple)
        assert len(entry.dtype) == 2
        d1, d2 = entry.dtype
        assert d1 is not None or d2 is not None
        # LEFT
        if d1 is not None:
            d1type, d1shape = eval_qdtype_node(d1)
            registers.append(
                Register(name=entry.name, dtype=d1type, side=Side.LEFT, shape=tuple(d1shape))
            )
        # RIGHT
        if d2 is not None:
            d2type, d2shape = eval_qdtype_node(d2)
            registers.append(
                Register(name=entry.name, dtype=d2type, side=Side.RIGHT, shape=tuple(d2shape))
            )

    sig = Signature(registers)
    logger.debug("Evaluated signature %s", sig)
    return sig


def eval_qdef_extern_node(qdef: QDefExternNode, *, safe: bool = True) -> 'qualtran.Bloq':
    """Evaluate an extern node by loading in the bloq using Python."""
    logger.info("Linking qdef extern %s from %s", qdef.bloq_key, qdef.cobject_from)
    if qdef.cobject_from is None:
        raise ValueError("The `from` clause is required for an extern qdef.")
    try:
        bloq = eval_cvalue_node(qdef.cobject_from, safe=safe)
    except (ValueError, ImportError, AttributeError, TypeError) as e:
        raise ValueError(*e.args, f'in {qdef}') from e
    return bloq


def eval_bloq_maybe_aliased(
    key: BloqKey,
    qdefs: Dict[BloqKey, QDefNode],
    qlocals: Mapping[BloqKey, Union[BloqKey, 'SoquetT']],
    bloqs: Dict[BloqKey, Bloq],
) -> 'qualtran.Bloq':
    """Recursively load bloqs.

    You must kick this off by calling `eval_qdef_impl_node`. Each QCallNode will try to eval the
    bloq it is calling via this function.

    There are two recursive cases and two base cases.
     - If the bloq has already been loaded, return it. (base-case: cache)
     - If `key` is an alias, resolve it and call this function on the resolved key. (recurse: alias)
     - If `key` points to an 'extern' qdef, load the bloq using Python (base-case: extern)
     - Otherwise, call `eval_qdef_impl_node`, which will recursively come back to this function
       for each QCallNode it contains. (recurse: impl).
    """
    if key in bloqs:
        return bloqs[key]

    if key in qlocals:
        alias = qlocals[key]
        if not isinstance(alias, str):
            raise ValueError(f"Expected a bloq key local varibale, but got {alias} for {key}")
        return eval_bloq_maybe_aliased(alias, qdefs, qlocals, bloqs)

    if key in qdefs:
        qdef = qdefs[key]
        if isinstance(qdef, QDefExternNode):
            bloq = eval_qdef_extern_node(qdef)
            bloqs[key] = bloq
            return bloq
        elif isinstance(qdef, QDefImplNode):
            bloq = eval_qdef_impl_node(qdef, qdefs, bloqs)
            bloqs[key] = bloq
            return bloq
        else:
            raise TypeError(f"Unknown qdef type {qdef}")

    if key not in qdefs:
        raise ValueError(f"Could not resolve {key}")

    raise TypeError(f"Could not resolve {key}")


def eval_qarg_value(val: NestedQArgValue, qlocals: Mapping[str, 'qualtran.SoquetT']):
    if isinstance(val, QArgValueNode):
        if val.idx:
            return cast(np.ndarray, qlocals[val.name])[val.idx]
        else:
            return qlocals[val.name]

    return [eval_qarg_value(v, qlocals) for v in val]


def eval_qarg_nodes(qargs: Sequence[QArgNode], qlocals: Mapping[str, 'qualtran.SoquetT']):
    qkwargs = {}
    for qarg in qargs:
        qkwargs[qarg.key] = eval_qarg_value(qarg.value, qlocals)
    return qkwargs


def _eval_qdef_impl_node(
    qdef: QDefImplNode, qdefs: Dict[BloqKey, QDefNode], bloqs: Dict[BloqKey, Bloq], *, safe: bool
) -> 'qualtran.CompositeBloq':
    """Evaluate a QDefImplNode, which defines a bloq implementation through a series of statements.

    This uses `qualtran.BloqBuilder` to build a `qualtran.CompositeBloq` by processing
    the statements in the qdef.

    This function will depth-first resolve-and-evaluate each "callee" bloq.

    Args:
        qdef: The node to evaluate.
        qdefs: A registry mapping bloq key to its un-evaluated qdef _or_ another bloq key.
            This function will add bloq-key-to-bloq-key alias mappings when evaluating
            AliasAssignmentNode statements.
        bloqs: A registry mapping bloq key to previously-evaluated bloqs.

    Returns:
        An evaluated qualtran.Bloq. The *caller* of this function is responsible for updating
            the `bloqs` cache.
    """
    logger.info("Evaluating qdef impl %s", qdef.bloq_key)

    signature: Signature = eval_qsignature(qdef.qsignature)
    qlocals: Mapping[str, Union['SoquetT', BloqKey]]
    bb, qlocals = BloqBuilder.from_signature(signature)
    subbloq_aliases: Dict[Bloq, str] = {}

    stmt: StatementNode
    cbloq: Optional[CompositeBloq] = None  # filled in on QReturnNode
    for stmt in qdef.body:
        logger.debug("STMT: %s", stmt)

        if isinstance(stmt, AliasAssignmentNode):
            qlocals[stmt.alias] = stmt.bloq_key  # type: ignore[assignment]
            bloq = eval_bloq_maybe_aliased(stmt.bloq_key, qdefs, qlocals, bloqs)
            subbloq_aliases[bloq] = stmt.alias

        elif isinstance(stmt, QCallNode):
            bloq = eval_bloq_maybe_aliased(stmt.bloq_key, qdefs, qlocals, bloqs)
            qkwargs = eval_qarg_nodes(stmt.qargs, qlocals)
            qrets = bb.add_t(bloq, **qkwargs)
            if len(qrets) != len(stmt.lvalues):
                raise BloqError(
                    f"Calling {stmt.bloq_key} gives {len(qrets)} return values, but {len(stmt.lvalues)} lvalues were written"
                )
            for qret, lval in zip(qrets, stmt.lvalues):
                qlocals[lval] = qret

        elif isinstance(stmt, QReturnNode):
            qkwargs = eval_qarg_nodes(stmt.ret_mapping, qlocals)
            cbloq = bb.finalize(**qkwargs)
            break

        else:
            raise ValueError(f"Bad stmt {stmt}")

    if cbloq is None:
        raise ValueError(f"{qdef.bloq_key} qdef impl lacks a `return` statement.")

    # TODO: This section of the function appends additional data onto the CompositeBloq.
    #       This should probably be done as part of the BloqBuilder.

    assert isinstance(cbloq, CompositeBloq)
    cbloq = attrs.evolve(
        cbloq, bloq_key=qdef.bloq_key
    )  # TODO: Include bloq_key and subbloq_aliases in BloqBuilder.

    if qdef.cobject_from is not None:
        try:
            bobj = eval_cvalue_node(qdef.cobject_from, safe=safe)
            if not isinstance(bobj, Bloq):
                raise ValueError(f"Expected to load a bloq, but got {bobj!r}")
            cbloq = attrs.evolve(
                cbloq, decomposed_from=bobj
            )  # TODO: Include decomposed_frm in BloqBuilder
        except Exception as e:  # pylint: disable=broad-except
            logger.error(
                "Failed to load the corresponding bloq 'from' for %s: %s", qdef.bloq_key, e
            )
    return cbloq


def eval_qdef_impl_node(
    qdef: QDefImplNode,
    qdefs: Dict[BloqKey, QDefNode],
    bloqs: Dict[BloqKey, Bloq],
    *,
    safe: bool = True,
) -> 'qualtran.CompositeBloq':
    # Wrapper to un-roll error messages.
    # See `_eval_qdef_impl_node` for details about this function.
    try:
        return _eval_qdef_impl_node(qdef=qdef, qdefs=qdefs, bloqs=bloqs, safe=safe)
    except Exception as e:
        logger.error("%s", e)
        logger.error("During evaluation of %s", qdef.bloq_key)
        raise e


def eval_module(m: L1Module, *, safe: bool = True) -> Dict[BloqKey, 'qualtran.Bloq']:
    """Evaluate a parsed L1Module.

    This will call `eval_qdef_impl_node` or `eval_qdef_extern_node` on each qdef in the
    L1Module. Note that `eval_qdef_impl_node` is recursive, so bloqs will be evaluated
    in a depth-first traversal.
    """

    qdefs: Dict[BloqKey, QDefNode] = {qdef.bloq_key: qdef for qdef in m.qdefs}
    bloqs: Dict[BloqKey, Bloq] = {}
    for qdef in m.qdefs:
        bk = qdef.bloq_key

        if bk in bloqs:
            logger.debug("Already loaded %s during recursion", bk)
            continue

        bloq: Bloq
        if isinstance(qdef, QDefImplNode):
            bloq = eval_qdef_impl_node(qdef, qdefs, bloqs, safe=safe)
            bloqs[bk] = bloq

        elif isinstance(qdef, QDefExternNode):
            bloq = eval_qdef_extern_node(qdef, safe=safe)
            bloqs[bk] = bloq

        else:
            raise ValueError(f'{qdef}')

    # Return `bloqs` ordered according to `m.qdefs`
    return {qdef.bloq_key: bloqs[qdef.bloq_key] for qdef in m.qdefs}
