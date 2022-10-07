from typing import Optional, Iterable

from cirq_qubitization.gate_with_registers import Register

from attrs import frozen


@frozen
class SplitRegister(Register):
    name: str
    bitsize: int

    def left_names(self) -> Iterable[str]:
        yield self.name

    def right_names(self) -> Iterable[str]:
        for i in range(self.bitsize):
            # TODO: name collisions? but we need it to be a kwarg
            yield f'{self.name}{i}'


@frozen
class JoinRegister(Register):
    name: str
    bitsize: int

    def left_names(self) -> Iterable[str]:
        for i in range(self.bitsize):
            # TODO: name collisions? but we need it to be a kwarg
            yield f'{self.name}{i}'

    def right_names(self) -> Iterable[str]:
        yield self.name


@frozen
class ApplyFRegister(Register):
    name: str
    bitsize: int
    out_name: str
    in_text: Optional[str] = None
    out_text: Optional[str] = None

    def left_names(self) -> Iterable[str]:
        yield self.name

    def right_names(self) -> Iterable[str]:
        yield self.out_name
