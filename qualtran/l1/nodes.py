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
from typing import Optional, Sequence, Union

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

    items: Sequence[CValueNode] = attrs.field(converter=tuple)

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
    cargs: Sequence[CArgNode] = attrs.field(converter=tuple)

    def canonical_str(self) -> str:
        carg_str = ', '.join(carg.canonical_str() for carg in self.cargs)
        if carg_str:
            return f'{self.name}({carg_str})'
        return self.name
