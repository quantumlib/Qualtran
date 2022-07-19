import abc
import dataclasses
from typing import Sequence, Dict, Iterable

import cirq


@dataclasses.dataclass(frozen=True)
class Register:
    name: str
    bitsize: int


class Registers:
    def __init__(self, registers: Iterable[Register]):
        self._registers = list(registers)
        self._register_dict = {r.name: r for r in registers}
        if len(self._registers) != len(self._register_dict):
            raise ValueError("Please provide unique register names.")

    def i(self, i: int):
        return self._registers[i]

    def __getitem__(self, name: str):
        return self._register_dict[name]

    def __iter__(self):
        yield from self._registers

    def split_qubits(self, qubits: Sequence[cirq.Qid]) -> Dict[str, Sequence[cirq.Qid]]:
        qubit_regs = {}
        base = 0
        for reg in self:
            qubit_regs[reg.name] = qubits[base : base + reg.bitsize]
            base += reg.bitsize
        return qubit_regs


class GateWithRegisters(cirq.Gate, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def registers(self) -> Registers:
        ...

    def _num_qubits_(self) -> int:
        return sum(reg.bitsize for reg in self.registers)

    @abc.abstractmethod
    def decompose_from_registers(
        self, **qubit_regs: Dict[str, Sequence[cirq.Qid]]
    ) -> cirq.OP_TREE:
        ...

    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        qubit_regs = self.registers.split_qubits(qubits)
        yield from self.decompose_from_registers(**qubit_regs)
