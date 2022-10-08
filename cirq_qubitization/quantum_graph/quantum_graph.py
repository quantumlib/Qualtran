from dataclasses import dataclass
from functools import cached_property
from typing import Union, Tuple, Optional

import numpy as np

from cirq_qubitization.quantum_graph.bloq import Bloq

from attrs import frozen, field


@frozen
class BloqInstance:
    bloq: Bloq
    i: int

    def __repr__(self):
        return f'{self.bloq!r}<{self.i}>'


class DanglingT:
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
    binst: Union[BloqInstance, DanglingT]
    reg_name: str
    idx: Tuple[int, ...] = field(converter=_to_tuple, default=tuple())


LeftDangle = DanglingT("LeftDangle")
RightDangle = DanglingT("RightDangle")

# attrs note: Slots come into play because of our use of cached_property??? TODO: figure out
@frozen(slots=False)
class Wire:
    left: Soquet
    right: Soquet

    def _left_shape(self) -> Optional[Tuple[int, ...]]:
        if not isinstance(self.left.binst, BloqInstance):
            return None

        # tricky: we use the left object's right shape (we're a wire)
        shape = self.left.binst.bloq.registers[self.left.reg_name].right_shape

        # hack off any width associated with indexing
        nw = len(self.left.idx)
        return shape[nw:]

    def _right_shape(self) -> Optional[Tuple[int, ...]]:
        if not isinstance(self.right.binst, BloqInstance):
            return None

        # tricky: we use the right object's left shape (we're a wire)
        shape = self.right.binst.bloq.registers[self.right.reg_name].left_shape
        nw = len(self.right.idx)
        return shape[nw:]

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        ls = self._left_shape()
        rs = self._right_shape()

        # FIXME: get the idx.

        if ls is not None and rs is not None:
            if ls != rs:
                raise ValueError(f"Invalid Wire {self}: shape mismatch: {ls} != {rs}")
            return ls

        if ls is not None:
            return ls

        if rs is not None:
            return rs

        raise ValueError(
            f"Invalid Wire {self}: either the left or right soquet must be non-dangling."
        )

    def __repr__(self):
        return f'{self.left!r} -> {self.right!r}'
