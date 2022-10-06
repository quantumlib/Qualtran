from typing import Optional

from cirq_qubitization.gate_with_registers import Register

from attrs import frozen


@frozen
class SplitRegister(Register):
    name: str
    bitsize: int


@frozen
class JoinRegister(Register):
    name: str
    bitsize: int


@frozen
class ApplyFRegister(Register):
    name: str
    bitsize: int
    out_name: str
    in_text: Optional[str] = None
    out_text: Optional[str] = None
