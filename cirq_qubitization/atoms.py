from dataclasses import dataclass
from functools import cached_property
from typing import Sequence, Union, List

import cirq

from cirq_qubitization import MultiTargetCSwap
from cirq_qubitization.gate_with_registers import GateWithRegisters, Registers

@dataclass(frozen=True)
class Split(GateWithRegisters):
    bitsize: int

    def decompose_from_registers(self, **qubit_regs: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        raise NotImplementedError()

    @cached_property
    def registers(self) -> Registers:
        return Registers.build(x=self.bitsize)


@dataclass(frozen=True)
class Join(GateWithRegisters):
    bitsize: int

    def decompose_from_registers(self, **qubit_regs: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        raise NotImplementedError()

    @cached_property
    def registers(self) -> Registers:
        return Registers.build(x=self.bitsize)
