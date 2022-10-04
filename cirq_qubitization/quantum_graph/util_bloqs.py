from dataclasses import dataclass
from functools import cached_property

from cirq_qubitization.gate_with_registers import Registers
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.fancy_registers import SplitRegister, JoinRegister


@dataclass(frozen=True)
class Split(Bloq):
    bitsize: int

    @cached_property
    def registers(self) -> Registers:
        return Registers([SplitRegister(name='sss', bitsize=self.bitsize)])


@dataclass(frozen=True)
class Join(Bloq):
    bitsize: int

    @cached_property
    def registers(self) -> Registers:
        return Registers([JoinRegister(name='jjj', bitsize=self.bitsize)])
