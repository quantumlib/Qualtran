from dataclasses import dataclass
from functools import cached_property
from typing import Union, Sequence, TYPE_CHECKING

from cirq_qubitization.gate_with_registers import Registers
from cirq_qubitization.quantum_graph.bloq import Bloq, NoCirqEquivalent
from cirq_qubitization.quantum_graph.fancy_registers import SplitRegister, JoinRegister

if TYPE_CHECKING:
    import cirq


@dataclass(frozen=True)
class Split(Bloq):
    bitsize: int

    @cached_property
    def registers(self) -> Registers:
        return Registers([SplitRegister(name='sss', bitsize=self.bitsize)])

    def on_registers(
        self, **qubit_regs: Union['cirq.Qid', Sequence['cirq.Qid']]
    ) -> 'cirq.GateOperation':
        raise NoCirqEquivalent()


@dataclass(frozen=True)
class Join(Bloq):
    bitsize: int

    @cached_property
    def registers(self) -> Registers:
        return Registers([JoinRegister(name='jjj', bitsize=self.bitsize)])

    def on_registers(
        self, **qubit_regs: Union['cirq.Qid', Sequence['cirq.Qid']]
    ) -> 'cirq.GateOperation':
        raise NoCirqEquivalent()
