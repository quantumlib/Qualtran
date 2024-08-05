#  Copyright 2023 Google LLC
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

"""Plumbing for bloq-to-bloq `Connection`s."""
from functools import cached_property
from typing import Tuple, TYPE_CHECKING, Union

from attrs import field, frozen

if TYPE_CHECKING:
    from qualtran import Bloq, Register


@frozen
class BloqInstance:
    """A unique instance of a Bloq within a `CompositeBloq`.

    Attributes:
        bloq: The `Bloq`.
        i: An arbitrary index to disambiguate this instance from other Bloqs of the same type
            within a `CompositeBloq`.
    """

    bloq: 'Bloq'
    i: int

    def __str__(self):
        return f'{self.bloq}<{self.i}>'

    def bloq_is(self, t) -> bool:
        """Helper method that does `isinstance(self.bloq, t)`.

        This is also defined on `DanglingT`, so using this method on `binst` is equivalent
        to:

        >>> not isinstance(binst, DanglingT) and isinstance(binst.bloq, t)
        """
        return isinstance(self.bloq, t)


class DanglingT:
    """The type of the singleton objects `LeftDangle` and `RightDangle`.

    These objects are placeholders for the `binst` field of a `Soquet` that represents
    an "external wire". We can consider `Soquets` of this type to represent input or
    output data of a `CompositeBloq`.
    """

    def __init__(self, name: str):
        self._name = name

    def __repr__(self):
        return self._name

    def bloq_is(self, t) -> bool:
        """DanglingT.bloq_is(...) is always False.

        This is to support convenient isinstance checking on binst.bloq where
        binst may be a `DanglingT`.
        """
        return False


def _to_tuple(x: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
    if isinstance(x, int):
        return (x,)
    return x


@frozen
class Soquet:
    """One half of a connection.

    Users should not construct these directly. They should be marshalled
    by a `BloqBuilder`.

    A `Soquet` acts as the node type in our quantum compute graph. It is a particular
    register (by name and optional index) on a particular `Bloq` instance.

    A `Soquet` can also be present in an external connection (i.e. represent an unconnected input
    or output) by setting the `binst` attribute to `LeftDangle` or `RightDangle`.

    Attributes:
        binst: The BloqInstance to which this soquet belongs.
        reg: The register that this soquet is an instance of.
        idx: Registers with non-empty `shape` attributes are multi-dimensional. A soquet
            is an explicitly indexed instantiation of one element of the multi-dimensional
            register.
    """

    binst: Union[BloqInstance, DanglingT]
    reg: 'Register'
    idx: Tuple[int, ...] = field(converter=_to_tuple, default=tuple())

    @idx.validator
    def _check_idx(self, attribute, value):
        if len(value) != len(self.reg.shape):
            raise ValueError(f"Bad index shape {value} for {self.reg}.")
        for i, shape in zip(value, self.reg.shape):
            if i >= shape:
                raise ValueError(f"Bad index {i} for {self.reg}.")

    def pretty(self) -> str:
        label = self.reg.name
        if len(self.idx) > 0:
            return f'{label}[{", ".join(str(i) for i in self.idx)}]'
        return label

    def __str__(self) -> str:
        return f'{self.binst}.{self.pretty()}'


LeftDangle = DanglingT("LeftDangle")
RightDangle = DanglingT("RightDangle")


def _singleton_error(self, x):
    raise ValueError("Do not instantiate a new DanglingT. Use `LeftDangle` or `RightDangle`.")


DanglingT.__init__ = _singleton_error  # type: ignore[method-assign]


@frozen
class Connection:
    """A connection between two `Soquet`s.

    Quantum data flows from left to right. The graph implied by a collection of `Connections`s
    is directed.
    """

    left: Soquet
    right: Soquet

    @cached_property
    def shape(self) -> int:
        ls = self.left.reg.bitsize
        rs = self.right.reg.bitsize

        if ls != rs:
            raise ValueError(f"Invalid Connection {self}: shape mismatch: {ls} != {rs}")
        return ls

    def __str__(self) -> str:
        return f'{self.left} -> {self.right}'
