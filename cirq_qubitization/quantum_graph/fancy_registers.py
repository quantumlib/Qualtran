import dataclasses
from typing import Optional

from cirq_qubitization.gate_with_registers import Register


@dataclasses.dataclass(frozen=True)
class SplitRegister(Register):
    pass


@dataclasses.dataclass(frozen=True)
class JoinRegister(Register):
    pass


@dataclasses.dataclass(frozen=True)
class ApplyFRegister(Register):
    out_name: str
    in_text: Optional[str] = None
    out_text: Optional[str] = None
