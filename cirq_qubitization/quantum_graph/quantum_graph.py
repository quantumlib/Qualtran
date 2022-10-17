from functools import cached_property
from typing import Union, Tuple, Optional

import numpy as np

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister

from attrs import frozen, field


@frozen
class BloqInstance:
    bloq: Bloq
    i: int


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
class Wire:
    binst: Union[BloqInstance, DanglingT]
    soq: FancyRegister
    idx: Tuple[int, ...] = field(converter=_to_tuple, default=tuple())


LeftDangle = DanglingT("LeftDangle")
RightDangle = DanglingT("RightDangle")

# attrs note: Slots come into play because of our use of cached_property??? TODO: figure out
@frozen(slots=False)
class Connection:
    left: Wire
    right: Wire

    def _left_shape(self) -> Optional[Tuple[int, ...]]:
        if not isinstance(self.left.binst, BloqInstance):
            return None

        # tricky: we use the left object's right shape (we're a wire)
        soq = self.left.soq
        shape = soq.wireshape + (soq.bitsize,)

        # hack off any width associated with indexing
        nw = len(self.left.idx)
        return shape[nw:]

    def _right_shape(self) -> Optional[Tuple[int, ...]]:
        if not isinstance(self.right.binst, BloqInstance):
            return None

        # tricky: we use the right object's left shape (we're a wire)
        soq = self.right.soq
        shape = soq.wireshape + (soq.bitsize,)
        nw = len(self.right.idx)
        return shape[nw:]

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        ls = self._left_shape()
        rs = self._right_shape()

        # FIXME: get the idx.

        if ls is not None and rs is not None:
            if ls != rs:
                # raise ValueError(f"Invalid Connection {self}: shape mismatch: {ls} != {rs}")
                print(f"Invalid Connection {self}: shape mismatch: {ls} != {rs}")
            return ls

        if ls is not None:
            return ls

        if rs is not None:
            return rs

        raise ValueError(
            f"Invalid Connection {self}: either the left or right soquet must be non-dangling."
        )
