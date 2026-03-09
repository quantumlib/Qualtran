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
"""The Qualtran-L1 AST Nodes."""

import abc
from typing import Optional, Sequence, Tuple, TypeAlias, Union

import attrs


class L1ASTNode(metaclass=abc.ABCMeta):
    """Every L1 AST Node inherits from this base class.

    For walking the AST, we use a visitor pattern.
    See `qualtran.l1.L1VisitorBase` and its descendants for more.
    If you introduce a new node type, you will likely need to add corresponding
    methods to the various visitor classes.
    """


class CValueNode(L1ASTNode, metaclass=abc.ABCMeta):
    """Nodes corresponding to classical values.

    This includes
     - LiteralNode
     - TupleNode
     - CObjectNode
    """

    @abc.abstractmethod
    def canonical_str(self): ...


@attrs.frozen
class LiteralNode(CValueNode):
    """A literal classical value."""

    value: Union[int, float, str]

    def canonical_str(self):
        return f'{self.value!r}'


@attrs.frozen
class TupleNode(CValueNode):
    """A sequence (tuple) of classical values."""

    items: Sequence[CValueNode] = attrs.field(converter=tuple[CValueNode])

    def canonical_str(self):
        if len(self.items) == 1:
            return f'({self.items[0].canonical_str()},)'
        item_str = ', '.join(i.canonical_str() for i in self.items)
        return f'({item_str})'


@attrs.frozen
class CArgNode(L1ASTNode):
    """A classical value optionally associated with a string key."""

    key: Optional[str]
    value: CValueNode

    def canonical_str(self) -> str:
        if self.key:
            return f'{self.key}={self.value.canonical_str()}'
        return self.value.canonical_str()


@attrs.frozen
class CObjectNode(CValueNode):
    """A classical 'object'.

    In Qualtran-L1, an 'object' is a container data structure with
     - an object name (analogous to a class name)
     - a sequence of contained classical values. Each may be given a string key.
    """

    name: str
    cargs: Sequence[CArgNode] = attrs.field(converter=tuple[CArgNode])

    def canonical_str(self) -> str:
        carg_str = ', '.join(carg.canonical_str() for carg in self.cargs)
        if carg_str:
            return f'{self.name}({carg_str})'
        return self.name


@attrs.frozen
class QDTypeNode(L1ASTNode):
    """A quantum data type, optionally with a shape."""

    dtype: CObjectNode
    shape: Optional[Sequence[int]]


@attrs.frozen
class QSignatureEntry(L1ASTNode):
    """A quantum signature entry."""

    name: str
    dtype: Union[QDTypeNode, Tuple[Optional[QDTypeNode], Optional[QDTypeNode]]]


class StatementNode(L1ASTNode, metaclass=abc.ABCMeta):
    """Nodes which can serve as statements in a qdef.

    This base class's implementors include:
     - `AliasAssignmentNode`
     - `QCallNode`
     - `QReturnNode`
    """


@attrs.frozen
class AliasAssignmentNode(StatementNode):
    """A statement that assigns `bloq_key` to `alias`."""

    alias: str
    bloq_key: str

    def __str__(self):
        return f'[AA] {self.alias} = {self.bloq_key}'


@attrs.frozen
class QArgValueNode(L1ASTNode):
    """A quantum argument (specifically the value, not the kv pair).

    In general, a quantum argument is referenced by a local variable name and optional
    indices into the local variable.
    """

    name: str
    idx: Sequence[int]


NestedQArgValue: TypeAlias = Union[QArgValueNode, Sequence['NestedQArgValue']]


@attrs.frozen
class QArgNode(L1ASTNode):
    """A quantum argument given as a key-value pair.

    During a quantum call, we provide quantum arguments (always by keyword). The value
    being passed can be a local variable, an indexed local variable, or an arbitrarily nested
    list of indexed local variables.
    """

    key: str
    value: NestedQArgValue  # TODO: turn all to tuple


@attrs.frozen
class QCallNode(StatementNode):
    """A statement that calls a quantum subroutine."""

    bloq_key: str
    lvalues: Sequence[str] = attrs.field(converter=tuple)
    qargs: Sequence[QArgNode] = attrs.field(converter=tuple)


@attrs.frozen
class QReturnNode(StatementNode):
    """A statement that returns from a subroutine."""

    ret_mapping: Sequence[QArgNode]


class QDefNode(L1ASTNode, metaclass=abc.ABCMeta):
    """Nodes that serve as 'qdefs'.

    This base class's implementors include:
     - `QDefImplNode`
     - `QDefExternNode`
    """

    @property
    @abc.abstractmethod
    def bloq_key(self) -> str: ...

    @property
    @abc.abstractmethod
    def qsignature(self) -> Sequence[QSignatureEntry]: ...

    @property
    @abc.abstractmethod
    def cobject_from(self) -> Optional[CObjectNode]: ...


@attrs.frozen
class QDefImplNode(QDefNode):
    """A qdef defining a quantum subroutine.

    Args:
        bloq_key: The bloq_key this qdef defines.
        qsignature: The qdef's signature
        body: The sequence of statements forming the qdef's body.
        cobject_from: Optional classical object in the "from" clause.

    ```qlt
    qdef bloq1 [ ..qsignature.. ]
    from ..cobject_from..
    { ..body.. }
    ```
    """

    bloq_key: str
    qsignature: Sequence[QSignatureEntry] = attrs.field(converter=tuple)
    body: Sequence[StatementNode] = attrs.field(converter=tuple)
    cobject_from: Optional[CObjectNode]


@attrs.frozen
class QDefExternNode(QDefNode):
    """A qdef declaring an external quantum subroutine.

    Args:
        bloq_key: The bloq_key this qdef declares.
        qsignature: The signature of the external subroutine.
        cobject_from: Manditory classical object in the "from" clause.

    ```qlt
    extern qdef X
    from qualtran.bloqs.basic_gates.XGate()
    [q: QBit()]
    ```
    """

    bloq_key: str
    qsignature: Sequence[QSignatureEntry] = attrs.field(converter=tuple)
    cobject_from: CObjectNode


@attrs.frozen
class L1Module(L1ASTNode):
    """A module consisting of a sequence of qdefs.

    In Qualtran-L1, a text file containing a module must begin with the
    string "# Qualtran-L1", followed by a newline, followed by a syntax
    version specifier "# {major}.{minor}.{patch}".

    Args:
        qdefs: The sequence of qdefs.

    ```qlt
    # Qualtran-L1
    # 1.0.0

    qdef bloq1 [x: QBit()] { ... }
    qdef bloq2 [...] {...}
    ```
    """

    qdefs: Sequence[QDefNode] = attrs.field(converter=tuple)
