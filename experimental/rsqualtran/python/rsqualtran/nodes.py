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
"""The Qualtran-L1 AST Nodes backed by Rust PyO3 classes."""

import abc
from typing import Optional, Sequence, Tuple, TypeAlias, Union

import attrs

from . import _rsqlt


class _NodeMeta(abc.ABCMeta):
    def __instancecheck__(cls, instance):
        if hasattr(cls, "_rsqlt_cls") and cls._rsqlt_cls is not None:
            if isinstance(cls._rsqlt_cls, tuple):
                if isinstance(instance, cls._rsqlt_cls):
                    return True
            elif isinstance(instance, cls._rsqlt_cls):
                return True
        return super().__instancecheck__(instance)

    def __subclasscheck__(cls, subclass):
        if hasattr(cls, "_rsqlt_cls") and cls._rsqlt_cls is not None:
            if isinstance(cls._rsqlt_cls, tuple):
                if issubclass(subclass, cls._rsqlt_cls):
                    return True
            elif issubclass(subclass, cls._rsqlt_cls):
                return True
        return super().__subclasscheck__(subclass)


class L1ASTNode(metaclass=_NodeMeta):
    """Every L1 AST Node inherits from this base class."""

    _rsqlt_cls = (
        _rsqlt.LiteralNode,
        _rsqlt.TupleNode,
        _rsqlt.CArgNode,
        _rsqlt.CObjectNode,
        _rsqlt.QDTypeNode,
        _rsqlt.QSignatureEntry,
        _rsqlt.AliasAssignmentNode,
        _rsqlt.QArgValueNode,
        _rsqlt.QArgNode,
        _rsqlt.QCallNode,
        _rsqlt.QReturnNode,
        _rsqlt.QDefImplNode,
        _rsqlt.QDefExternNode,
        _rsqlt.QCastNode,
        _rsqlt.L1Module,
    )


class CValueNode(L1ASTNode, metaclass=_NodeMeta):
    """Nodes corresponding to classical values."""

    _rsqlt_cls = (_rsqlt.LiteralNode, _rsqlt.TupleNode, _rsqlt.CObjectNode)

    @abc.abstractmethod
    def canonical_str(self): ...


def _literal_canonical_str(self):
    return f"{self.value!r}"


def _tuple_canonical_str(self):
    if len(self.items) == 1:
        return f"({self.items[0].canonical_str()},)"
    item_str = ", ".join(i.canonical_str() for i in self.items)
    return f"({item_str})"


def _carg_canonical_str(self) -> str:
    if self.key:
        return f"{self.key}={self.value.canonical_str()}"
    return self.value.canonical_str()


def _cobject_canonical_str(self) -> str:
    carg_str = ", ".join(carg.canonical_str() for carg in self.cargs)
    if carg_str:
        return f"{self.name}({carg_str})"
    return self.name


_rsqlt.LiteralNode.canonical_str = _literal_canonical_str
_rsqlt.TupleNode.canonical_str = _tuple_canonical_str
_rsqlt.CArgNode.canonical_str = _carg_canonical_str
_rsqlt.CObjectNode.canonical_str = _cobject_canonical_str
_rsqlt.QCastNode.cobject_from = property(lambda self: None)


class LiteralNode(CValueNode):
    """A literal classical value."""

    _rsqlt_cls = _rsqlt.LiteralNode

    def __new__(cls, value: Union[int, float, str]):
        if not isinstance(value, (int, float, str)):
            raise TypeError(f"LiteralNode value must be int, float, or str, got {type(value)}")
        return _rsqlt.LiteralNode(value)


class TupleNode(CValueNode):
    """A sequence (tuple) of classical values."""

    _rsqlt_cls = _rsqlt.TupleNode

    def __new__(cls, items: Sequence[CValueNode]):
        items_list = list(items)
        for item in items_list:
            if not isinstance(item, CValueNode._rsqlt_cls):
                raise TypeError(f"TupleNode items must be CValueNode, got {type(item)}")
        return _rsqlt.TupleNode(items_list)


class CArgNode(L1ASTNode):
    """A classical value optionally associated with a string key."""

    _rsqlt_cls = _rsqlt.CArgNode

    def __new__(cls, key: Optional[str], value: CValueNode):
        if key is not None and not isinstance(key, str):
            raise TypeError(f"CArgNode key must be str or None, got {type(key)}")
        if not isinstance(value, CValueNode._rsqlt_cls):
            raise TypeError(f"CArgNode value must be CValueNode, got {type(value)}")
        return _rsqlt.CArgNode(value, key=key)


class CObjectNode(CValueNode):
    """A classical 'object'."""

    _rsqlt_cls = _rsqlt.CObjectNode

    def __new__(cls, name: str, cargs: Sequence[CArgNode]):
        if not isinstance(name, str):
            raise TypeError(f"CObjectNode name must be str, got {type(name)}")
        cargs_list = list(cargs)
        for carg in cargs_list:
            if not isinstance(carg, _rsqlt.CArgNode):
                raise TypeError(f"CObjectNode cargs must be CArgNode, got {type(carg)}")
        return _rsqlt.CObjectNode(name, cargs_list)


class QDTypeNode(L1ASTNode):
    """A quantum data type, optionally with a shape."""

    _rsqlt_cls = _rsqlt.QDTypeNode

    def __new__(cls, dtype: CObjectNode, shape: Optional[Sequence[int]]):
        if not isinstance(dtype, _rsqlt.CObjectNode):
            raise TypeError(f"QDTypeNode dtype must be CObjectNode, got {type(dtype)}")
        shape_list = list(shape) if shape is not None else None
        if shape_list is not None:
            for s in shape_list:
                if not isinstance(s, int):
                    raise TypeError(f"QDTypeNode shape must be sequence of int, got {type(s)}")
        return _rsqlt.QDTypeNode(dtype, shape=shape_list)


class QSignatureEntry(L1ASTNode):
    """A quantum signature entry."""

    _rsqlt_cls = _rsqlt.QSignatureEntry

    def __new__(
        cls,
        name: str,
        dtype: Union[QDTypeNode, Tuple[Optional[QDTypeNode], Optional[QDTypeNode]]],
        annotation: Optional[CValueNode] = None,
    ):
        if not isinstance(name, str):
            raise TypeError(f"QSignatureEntry name must be str, got {type(name)}")
        if not isinstance(dtype, _rsqlt.QDTypeNode) and not isinstance(dtype, tuple):
            raise TypeError(f"QSignatureEntry dtype must be QDTypeNode or tuple, got {type(dtype)}")
        if isinstance(dtype, tuple):
            if len(dtype) != 2:
                raise ValueError(
                    f"QSignatureEntry dtype tuple must have length 2, got {len(dtype)}"
                )
            if dtype[0] is not None and not isinstance(dtype[0], _rsqlt.QDTypeNode):
                raise TypeError(
                    f"QSignatureEntry dtype[0] must be QDTypeNode or None, got {type(dtype[0])}"
                )
            if dtype[1] is not None and not isinstance(dtype[1], _rsqlt.QDTypeNode):
                raise TypeError(
                    f"QSignatureEntry dtype[1] must be QDTypeNode or None, got {type(dtype[1])}"
                )
        if annotation is not None and not isinstance(annotation, CValueNode._rsqlt_cls):
            raise TypeError(
                f"QSignatureEntry annotation must be CValueNode or None, got {type(annotation)}"
            )

        return _rsqlt.QSignatureEntry(name, dtype)


_orig_evolve = attrs.evolve


def _custom_evolve(inst, **changes):
    if isinstance(inst, _rsqlt.QSignatureEntry):
        name = changes.get("name", inst.name)
        dtype = changes.get("dtype", inst.dtype)
        annotation = changes.get("annotation", None)
        return QSignatureEntry(name, dtype, annotation=annotation)
    return _orig_evolve(inst, **changes)


attrs.evolve = _custom_evolve


@attrs.frozen
class QStructEntry(L1ASTNode):
    """A qstruct entry."""

    _rsqlt_cls = None

    name: str
    dtype: QDTypeNode


@attrs.frozen
class LValueNode(L1ASTNode):
    """An l-value with an optional annotation."""

    _rsqlt_cls = None

    name: str
    annotation: Optional[CValueNode] = None

    def __str__(self):
        if self.annotation:
            return f"{self.name} @ {self.annotation.canonical_str()}"
        return self.name


class StatementNode(L1ASTNode, metaclass=_NodeMeta):
    """Nodes which can serve as statements in a qdef."""

    _rsqlt_cls = (_rsqlt.AliasAssignmentNode, _rsqlt.QCallNode, _rsqlt.QReturnNode)


class AliasAssignmentNode(StatementNode):
    """A statement that assigns `bloq_key` to `alias`."""

    _rsqlt_cls = _rsqlt.AliasAssignmentNode

    def __new__(cls, alias: str, bloq_key: str):
        if not isinstance(alias, str):
            raise TypeError(f"AliasAssignmentNode alias must be str, got {type(alias)}")
        if not isinstance(bloq_key, str):
            raise TypeError(f"AliasAssignmentNode bloq_key must be str, got {type(bloq_key)}")
        return _rsqlt.AliasAssignmentNode(alias, bloq_key)


class QArgValueNode(L1ASTNode):
    """A quantum argument (specifically the value, not the kv pair)."""

    _rsqlt_cls = _rsqlt.QArgValueNode

    def __new__(cls, name: str, idx: Sequence[int]):
        if not isinstance(name, str):
            raise TypeError(f"QArgValueNode name must be str, got {type(name)}")
        idx_list = list(idx)
        for i in idx_list:
            if not isinstance(i, int):
                raise TypeError(f"QArgValueNode idx must be sequence of int, got {type(i)}")
        return _rsqlt.QArgValueNode(name, idx_list)


NestedQArgValue: TypeAlias = Union[QArgValueNode, Sequence["NestedQArgValue"]]


class QArgNode(L1ASTNode):
    """A quantum argument given as a key-value pair."""

    _rsqlt_cls = _rsqlt.QArgNode

    def __new__(cls, key: str, value: NestedQArgValue, annotation: Optional[CValueNode] = None):
        if not isinstance(key, str):
            raise TypeError(f"QArgNode key must be str, got {type(key)}")
        if annotation is not None and not isinstance(annotation, CValueNode._rsqlt_cls):
            raise TypeError(
                f"QArgNode annotation must be CValueNode or None, got {type(annotation)}"
            )

        def _convert_nested(v):
            if isinstance(v, _rsqlt.QArgValueNode):
                return v
            if isinstance(v, (list, tuple)):
                return [_convert_nested(x) for x in v]
            raise TypeError(
                f"QArgNode value must be QArgValueNode or nested sequence, got {type(v)}"
            )

        val_converted = _convert_nested(value)
        return _rsqlt.QArgNode(key, val_converted)


class QCallNode(StatementNode):
    """A statement that calls a quantum subroutine."""

    _rsqlt_cls = _rsqlt.QCallNode

    def __new__(
        cls,
        bloq_key: str,
        lvalues: Sequence[Union[LValueNode, str]],
        qargs: Sequence[QArgNode],
        annotation: Optional[CValueNode] = None,
    ):
        if not isinstance(bloq_key, str):
            raise TypeError(f"QCallNode bloq_key must be str, got {type(bloq_key)}")

        lvalues_list = []
        for lv in lvalues:
            if isinstance(lv, LValueNode):
                lvalues_list.append(lv.name)
            elif isinstance(lv, str):
                lvalues_list.append(lv)
            else:
                raise TypeError(f"QCallNode lvalues must be LValueNode or str, got {type(lv)}")

        qargs_list = list(qargs)
        for qa in qargs_list:
            if not isinstance(qa, _rsqlt.QArgNode):
                raise TypeError(f"QCallNode qargs must be QArgNode, got {type(qa)}")

        if annotation is not None and not isinstance(annotation, CValueNode._rsqlt_cls):
            raise TypeError(
                f"QCallNode annotation must be CValueNode or None, got {type(annotation)}"
            )

        return _rsqlt.QCallNode(bloq_key, lvalues_list, qargs_list)


class QReturnNode(StatementNode):
    """A statement that returns from a subroutine."""

    _rsqlt_cls = _rsqlt.QReturnNode

    def __new__(cls, ret_mapping: Sequence[QArgNode]):
        ret_list = list(ret_mapping)
        for qa in ret_list:
            if not isinstance(qa, _rsqlt.QArgNode):
                raise TypeError(f"QReturnNode ret_mapping must be QArgNode, got {type(qa)}")
        return _rsqlt.QReturnNode(ret_list)


class QDefNode(L1ASTNode, metaclass=_NodeMeta):
    """Nodes that serve as 'qdefs'."""

    _rsqlt_cls = (_rsqlt.QDefImplNode, _rsqlt.QDefExternNode, _rsqlt.QCastNode)

    @property
    @abc.abstractmethod
    def bloq_key(self) -> str: ...

    @property
    @abc.abstractmethod
    def qsignature(self) -> Sequence[QSignatureEntry]: ...

    @property
    @abc.abstractmethod
    def cobject_from(self) -> Optional[CObjectNode]: ...


class QDefImplNode(QDefNode):
    """A qdef defining a quantum subroutine."""

    _rsqlt_cls = _rsqlt.QDefImplNode

    def __new__(
        cls,
        bloq_key: str,
        qsignature: Sequence[QSignatureEntry],
        body: Sequence[StatementNode],
        cobject_from: Optional[CObjectNode],
    ):
        if not isinstance(bloq_key, str):
            raise TypeError(f"QDefImplNode bloq_key must be str, got {type(bloq_key)}")
        qsig_list = list(qsignature)
        for qs in qsig_list:
            if not isinstance(qs, _rsqlt.QSignatureEntry):
                raise TypeError(f"QDefImplNode qsignature must be QSignatureEntry, got {type(qs)}")
        body_list = list(body)
        for stmt in body_list:
            if not isinstance(stmt, StatementNode._rsqlt_cls):
                raise TypeError(f"QDefImplNode body must be StatementNode, got {type(stmt)}")
        if cobject_from is not None and not isinstance(cobject_from, _rsqlt.CObjectNode):
            raise TypeError(
                f"QDefImplNode cobject_from must be CObjectNode or None, got {type(cobject_from)}"
            )

        return _rsqlt.QDefImplNode(bloq_key, qsig_list, body_list, cobject_from=cobject_from)


class QDefExternNode(QDefNode):
    """A qdef declaring an external quantum subroutine."""

    _rsqlt_cls = _rsqlt.QDefExternNode

    def __new__(
        cls,
        bloq_key: str,
        qsignature: Sequence[QSignatureEntry],
        cobject_from: Optional[CObjectNode],
    ):
        if not isinstance(bloq_key, str):
            raise TypeError(f"QDefExternNode bloq_key must be str, got {type(bloq_key)}")
        qsig_list = list(qsignature)
        for qs in qsig_list:
            if not isinstance(qs, _rsqlt.QSignatureEntry):
                raise TypeError(
                    f"QDefExternNode qsignature must be QSignatureEntry, got {type(qs)}"
                )
        if cobject_from is not None and not isinstance(cobject_from, _rsqlt.CObjectNode):
            raise TypeError(
                f"QDefExternNode cobject_from must be CObjectNode or None, got {type(cobject_from)}"
            )
        if cobject_from is None:
            raise ValueError("QDefExternNode cobject_from cannot be None in rsqualtran")

        return _rsqlt.QDefExternNode(bloq_key, qsig_list, cobject_from)


class QCastNode(QDefNode):
    """A qcast declaring a casting (bookkeeping) operation."""

    _rsqlt_cls = _rsqlt.QCastNode

    def __new__(cls, bloq_key: str, qsignature: Sequence[QSignatureEntry]):
        if not isinstance(bloq_key, str):
            raise TypeError(f"QCastNode bloq_key must be str, got {type(bloq_key)}")
        qsig_list = list(qsignature)
        for qs in qsig_list:
            if not isinstance(qs, _rsqlt.QSignatureEntry):
                raise TypeError(f"QCastNode qsignature must be QSignatureEntry, got {type(qs)}")

        return _rsqlt.QCastNode(bloq_key, qsig_list)

    @property
    def cobject_from(self) -> Optional[CObjectNode]:
        return None


@attrs.frozen
class QStructNode(L1ASTNode):
    """A qstruct type definition."""

    _rsqlt_cls = None

    symbol_id: str
    qfields: Sequence[QStructEntry] = attrs.field(converter=tuple[QStructEntry])
    cobject_from: Optional[CObjectNode]


class L1Module(L1ASTNode):
    """A module consisting of a sequence of qdefs."""

    _rsqlt_cls = _rsqlt.L1Module

    def __new__(cls, qdefs: Sequence[QDefNode], qstructs: Sequence[QStructNode] = ()):
        qdefs_list = list(qdefs)
        for qd in qdefs_list:
            if not isinstance(qd, QDefNode._rsqlt_cls):
                raise TypeError(f"L1Module qdefs must be QDefNode, got {type(qd)}")
        qstructs_list = list(qstructs)
        for qs in qstructs_list:
            if not isinstance(qs, QStructNode):
                raise TypeError(f"L1Module qstructs must be QStructNode, got {type(qs)}")

        return _rsqlt.L1Module(qdefs_list)
