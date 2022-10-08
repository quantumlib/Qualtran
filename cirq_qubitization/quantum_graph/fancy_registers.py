from typing import Optional, Iterable, Tuple

from cirq_qubitization.gate_with_registers import Register, IThruRegister

from attrs import frozen


@frozen
class SplitRegister(Register):
    name: str
    n: int

    @property
    def left_shape(self) -> Tuple[int, ...]:
        return (self.n,)

    @property
    def right_shape(self) -> Tuple[int, ...]:
        return (self.n, 1)


@frozen
class JoinRegister(Register):
    name: str
    n: int

    @property
    def left_shape(self) -> Tuple[int, ...]:
        return (self.n, 1)

    @property
    def right_shape(self) -> Tuple[int, ...]:
        return (self.n,)


@frozen
class ApplyFRegister(IThruRegister):
    name: str
    bitsize: int
    out_name: str
    in_text: Optional[str] = None
    out_text: Optional[str] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return (self.bitsize,)
