from typing import Union

from attrs import frozen

from cirq_qubitization.quantum_graph.bloq import Bloq


@frozen
class BloqInstance:
    """A unique instance of a Bloq within a `CompositeBloq`.

    Attributes:
        bloq: The `Bloq`.
        i: An arbitary index to disambiguate this instance from other Bloqs of the same type
            within a `CompositeBloq`.
    """

    bloq: Bloq
    i: int


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

    # def __eq__(self, other):
    #     raise ValueError("Do not use equality comparison on DanglingT. Use `is`.")


@frozen
class Soquet:
    """One half of a `Wire` connection.

    A `Soquet` acts as the node type in our quantum compute graph. It is a particular
    register (by name) on a particular `Bloq`.

    A `Soquet can also be present in a dangling wire (i.e. represent an unconnected input or
    output) by setting the `binst` attribute to `LeftDangle` or `RightDangle`.
    """

    binst: Union[BloqInstance, DanglingT]
    reg_name: str


LeftDangle = DanglingT("LeftDangle")
RightDangle = DanglingT("RightDangle")


def _singleton_error(self, x):
    raise ValueError("Do not instantiate a new DanglingT. Use `LeftDangle` or `RightDangle`.")


DanglingT.__init__ = _singleton_error


@frozen
class Wire:
    """A connection between two `Soquet`s.

    Quantum data flows from left to right. The graph implied by a collection of `Wire`s
    is directed.
    """

    left: Soquet
    right: Soquet
