from dataclasses import dataclass
from typing import Union

from cirq_qubitization.quantum_graph.bloq import Bloq


@dataclass(frozen=True)
class BloqInstance:
    bloq: Bloq
    i: int

    def __repr__(self):
        return f'{self.bloq!r}<{self.i}>'


class DanglingT:
    def __init__(self, direction: str):
        self.direction = direction


@dataclass(frozen=True)
class Port:
    binst: Union[BloqInstance, DanglingT]
    reg_name: str


LeftDangle = DanglingT(direction='l')
RightDangle = DanglingT(direction='r')


@dataclass(frozen=True)
class Wire:
    left: Port
    right: Port

    def __repr__(self):
        return f'{self.left!r} -> {self.right!r}'
