from typing import Tuple, Union

from attrs import field, frozen

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister


@frozen
class BloqInstance:
    """A unique instance of a Bloq within a `CompositeBloq`.

    Attributes:
        bloq: The `Bloq`.
        i: An arbitrary index to disambiguate this instance from other Bloqs of the same type
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


def _to_tuple(x: Union[int, Tuple[int, ...]]):
    if isinstance(x, int):
        return (x,)
    return x


@frozen
class Soquet:
    """One half of a connection.

    Users should not construct these directly. They should be marshalled
    by a `CompositeBloqBuilder`.

    A `Soquet` acts as the node type in our quantum compute graph. It is a particular
    register (by name and optional index) on a particular `Bloq` instance.

    A `Soquet` can also be present in an external connection (i.e. represent an unconnected input
    or output) by setting the `binst` attribute to `LeftDangle` or `RightDangle`.

    Args:
        binst: The BloqInstance to which this soquet belongs.
        reg: The register that this soquet is an instance of.
        idx: Registers with non-empty `wireshape` attributes are multi-dimensional. A soquet
            is an explicitly indexed instantiation of one element of the multi-dimensional
            register.
    """

    binst: Union[BloqInstance, DanglingT]
    reg: FancyRegister
    idx: Tuple[int, ...] = field(converter=_to_tuple, default=tuple())

    @idx.validator
    def _check_idx(self, attribute, value):
        if len(value) != len(self.reg.wireshape):
            raise ValueError(f"Bad index shape {value} for {self.reg}.")
        for i, shape in zip(value, self.reg.wireshape):
            if i >= shape:
                raise ValueError(f"Bad index {i} for {self.reg}.")

    def pretty(self) -> str:
        label = self.reg.name
        if len(self.idx) > 0:
            return f'{label}[{", ".join(str(i) for i in self.idx)}]'
        return label


LeftDangle = DanglingT("LeftDangle")
RightDangle = DanglingT("RightDangle")


def _singleton_error(self, x):
    raise ValueError("Do not instantiate a new DanglingT. Use `LeftDangle` or `RightDangle`.")


DanglingT.__init__ = _singleton_error


@frozen
class Connection:
    """A connection between two `Soquet`s.

    Quantum data flows from left to right. The graph implied by a collection of `Connections`s
    is directed.
    """

    left: Soquet
    right: Soquet
